using GGMLSharp;
using ModelLoader;
using static GGMLSharp.Structs;

namespace MNIST_CPU
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Model model = LoadModel("./Assets/mnist_model.state_dict");
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

			SafeGGmlTensor probs = Compute(model);
			byte[] data = probs.GetBackend();
			List<float> probsList = DataConverter.ConvertToFloats(data).ToList();
			int prediction = probsList.IndexOf(probsList.Max());
			model.context.Free();

			return prediction;
		}


		public class Model
		{
			public SafeGGmlTensor input;
			public SafeGGmlTensor fc2Weight;
			public SafeGGmlTensor fc2Bias;
			public SafeGGmlTensor fc1Weight;
			public SafeGGmlTensor fc1Bias;
			public SafeGGmlContext context;
			public SafeGGmlBackend backend;
			public SafeGGmlGraph graph;
			public SafeGGmlBackendBuffer buffer;
			public SafeGGmlGraphAllocr allocr;
		}

		public static Model LoadModel(string path)
		{
			PickleLoader pickleLoader = new PickleLoader();
			List<Tensor> tensors = pickleLoader.ReadTensorsInfoFromFile(path);
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
			model.input = model.context.NewTensor1d(GGmlType.GGML_TYPE_F32, 28 * 28);

			var t = tensors.Find(a => a.Name == "fc1.weight");
			model.fc1Weight = model.context.NewTensor(t.Type, t.Shape.ToArray());

			t = tensors.Find(a => a.Name == "fc1.bias");
			model.fc1Bias = model.context.NewTensor(t.Type, t.Shape.ToArray());

			t = tensors.Find(a => a.Name == "fc2.weight");
			model.fc2Weight = model.context.NewTensor(t.Type, t.Shape.ToArray());

			t = tensors.Find(a => a.Name == "fc2.bias");
			model.fc2Bias = model.context.NewTensor(t.Type, t.Shape.ToArray());

			model.buffer = model.context.BackendAllocContextTensors(model.backend);

			model.fc1Weight.SetBackend(pickleLoader.ReadByteFromFile(tensors.First(a => a.Name == "fc1.weight")));
			model.fc1Bias.SetBackend(pickleLoader.ReadByteFromFile(tensors.First(a => a.Name == "fc1.bias")));
			model.fc2Weight.SetBackend(pickleLoader.ReadByteFromFile(tensors.First(a => a.Name == "fc2.weight")));
			model.fc2Bias.SetBackend(pickleLoader.ReadByteFromFile(tensors.First(a => a.Name == "fc2.bias")));


			return model;
		}

		private static void BuildGraph(Model model)
		{
			model.graph = model.context.NewGraph();

			SafeGGmlTensor re = model.context.Linear(model.input, model.context.Reshape2d(model.fc1Weight, model.fc1Weight.Shape[1], model.fc1Weight.Shape[0]), model.fc1Bias);
			re = model.context.Relu(re);
			re = model.context.Linear(re, model.context.Reshape2d(model.fc2Weight, model.fc2Weight.Shape[1], model.fc2Weight.Shape[0]), model.fc2Bias);
			re = model.context.SoftMax(re);
			re.Name = "probs";

			model.graph.BuildForwardExpend(re);
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
