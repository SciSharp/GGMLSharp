using GGMLSharp;
using static GGMLSharp.Structs;

namespace MNIST_CNN
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Model model = LoadModel("./Assets/mnist-cnn-model.gguf");
			byte[] bytes = File.ReadAllBytes("./Assets/image.raw");
			Console.WriteLine("The image is:");
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					Console.Write(bytes[i * 28 + j] > 200 ? " " : "*");
				}
				Console.WriteLine();
			}

			float[] digit = new float[28 * 28];
			for (int i = 0; i < bytes.Length; i++)
			{
				digit[i] = bytes[i] / 255.0f;
			}

			int prediction = Eval(model, digit);
			Console.WriteLine("Prediction: {0}", prediction);
			Console.ReadKey();
		}

		private static int Eval(Model model, float[] digit)
		{
			model.input.SetBackend(digit);
			// calculate the temporaly memory required to compute
			model.allocr = new SafeGGmlGraphAllocr(model.buffer.BufferType);

			// create the worst case graph for memory usage estimation
			BuildGraph(model);
			model.graph.Reserve(model.allocr);
			ulong mem_size = model.allocr.GetBufferSize(0);
			Console.WriteLine($"compute buffer size: {mem_size / 1024.0} KB");
			SafeGGmlTensor probs =  Compute(model);
			byte[] data = probs.GetBackend();
			List<float> probsList = DataConverter.ConvertToFloats(data).ToList();
			int prediction = probsList.IndexOf(probsList.Max());
			model.context.Free();

			return prediction;
		}


		public class Model
		{
			public SafeGGmlTensor input;
			public SafeGGmlTensor conv2d1Kernel;
			public SafeGGmlTensor conv2d1Bias;
			public SafeGGmlTensor conv2d2Kernel;
			public SafeGGmlTensor conv2d2Bias;
			public SafeGGmlTensor denseWeight;
			public SafeGGmlTensor denseBias;
			public SafeGGmlContext context;
			public SafeGGmlBackend backend;
			public SafeGGmlBackendBuffer buffer;
			public SafeGGmlGraph graph;
			public SafeGGmlGraphAllocr allocr;
		}

		public static Model LoadModel(string path)
		{
			SafeGGmlContext ggmlCtx = new SafeGGmlContext();
			SafeGGufContext ggufCtx = SafeGGufContext.InitFromFile(path, ggmlCtx, false);
			Model model = new Model();

			if (SafeGGmlBackend.HasCuda)
			{
				model.backend = SafeGGmlBackend.CudaInit(); // init device 0
			}
			else
			{
				model.backend = SafeGGmlBackend.CpuInit();
			}

			if (model.backend == null)
			{
				Console.WriteLine("ggml_backend_cuda_init() failed.");
				Console.WriteLine("we while use ggml_backend_cpu_init() instead.");

				// if there aren't GPU Backends fallback to CPU backend
				model.backend = SafeGGmlBackend.CpuInit();
			}

			model.context = new SafeGGmlContext(IntPtr.Zero, NoAllocateMemory: true);
			model.input = model.context.NewTensor2d(GGmlType.GGML_TYPE_F32, 28, 28);

			SafeGGmlTensor t = ggmlCtx.GetTensor("kernel1");
			model.conv2d1Kernel = model.context.NewTensor(t.Type, t.Shape);

			t = ggmlCtx.GetTensor("bias1");
			model.conv2d1Bias = model.context.NewTensor(t.Type, t.Shape);

			t = ggmlCtx.GetTensor("kernel2");
			model.conv2d2Kernel = model.context.NewTensor(t.Type, t.Shape);

			t = ggmlCtx.GetTensor("bias2");
			model.conv2d2Bias = model.context.NewTensor(t.Type, t.Shape);

			t = ggmlCtx.GetTensor("dense_w");
			model.denseWeight = model.context.NewTensor(t.Type, t.Shape);

			t = ggmlCtx.GetTensor("dense_b");
			model.denseBias = model.context.NewTensor(t.Type, t.Shape);

			model.buffer = model.context.BackendAllocContextTensors(model.backend);

			byte[] bytes = ggmlCtx.GetTensor("kernel1").GetData();
			model.conv2d1Kernel.SetBackend(bytes);

			bytes = ggmlCtx.GetTensor("bias1").GetData();
			model.conv2d1Bias.SetBackend(bytes);

			bytes = ggmlCtx.GetTensor("kernel2").GetData();
			model.conv2d2Kernel.SetBackend(bytes);

			bytes = ggmlCtx.GetTensor("bias2").GetData();
			model.conv2d2Bias.SetBackend(bytes);

			bytes = ggmlCtx.GetTensor("dense_w").GetData();
			model.denseWeight.SetBackend(bytes);

			bytes = ggmlCtx.GetTensor("dense_b").GetData();
			model.denseBias.SetBackend(bytes);

			return model;
		}

		private static void BuildGraph(Model model)
		{
			model.graph = model.context.NewGraph();

			SafeGGmlTensor cur = model.context.Conv2d(model.input, model.conv2d1Kernel, model.conv2d1Bias);
			cur = model.context.Relu(cur);
			// Output shape after Conv2D: (26 26 32 1)
			cur = model.context.Pool2d(cur);
			// Output shape after MaxPooling2D: (13 13 32 1)
			cur = model.context.Conv2d(cur, model.conv2d2Kernel, model.conv2d2Bias);

			cur = model.context.Relu(cur);
			// Output shape after Conv2D: (11 11 64 1)
			cur = model.context.Pool2d(cur);
			// Output shape after MaxPooling2D: (5 5 64 1)
			cur = model.context.Permute(cur, 1, 2, 0, 3);
			cur = model.context.Cont(cur);
			// Output shape after permute: (64 5 5 1)
			cur = model.context.Reshape2d(cur, 1600, 1);
			// Final Dense layer
			cur = model.context.Linear(cur, model.denseWeight, model.denseBias);
			cur = model.context.SoftMax(cur);
			cur.Name = "probs";

			model.graph.BuildForwardExpend(cur);

		}

		// compute with backend
		private static SafeGGmlTensor Compute(Model model)
		{
			// allocate tensors
			model.graph.GraphAllocate(model.allocr);

			model.graph.BackendCompute(model.backend);

			// in this case, the output tensor is the last one in the graph
			return model.graph.Nodes[model.graph.NodeCount - 1];
		}


	}
}
