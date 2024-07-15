using GGMLSharp;
using ModelLoader;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static GGMLSharp.Structs;

namespace MNIST_CPU
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Model model = LoadModel(@".\Assets\mnist_model.state_dict");
			byte[] bytes = File.ReadAllBytes(@".\Assets\image.raw");
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

			int prediction = Eval(model, 1, digit);
			Console.WriteLine("Prediction: {0}", prediction);
			Console.ReadKey();
		}

		private static int Eval(Model model, int n_threads, float[] digit)
		{
			model.input.SetBackend(digit);
			// calculate the temporaly memory required to compute
			SafeGGmlGraphAllocr allocr = new SafeGGmlGraphAllocr(model.backend.GetDefaultBufferType());

			// create the worst case graph for memory usage estimation
			BuildGraph(model);
			model.graph.Reserve(allocr);
			ulong mem_size = allocr.GetBufferSize(0);
			Console.WriteLine($"compute buffer size: {mem_size / 1024.0} KB");

			SafeGGmlTensor probs = Compute(model, allocr);
			List<float> probsList = probs.GetDataInFloats().ToList(); ;
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
			public SafeGGmlBackendBuffer buffer;
			public SafeGGmlGraph graph;
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
			model.input.Name = "input";

			model.buffer = model.context.BackendAllocContextTensors(model.backend);

			foreach (var tensor in tensors)
			{
				if (tensor.Name == "fc2.weight")
				{
					model.fc2Weight = LoadWeigth(model.context, pickleLoader, tensor);
					model.fc2Weight.Name = "fc2.weight";
				}
				else if (tensor.Name == "fc2.bias")
				{
					model.fc2Bias = LoadWeigth(model.context, pickleLoader, tensor);
					model.fc2Bias.Name = "fc2.bias";
				}
				else if (tensor.Name == "fc1.weight")
				{
					model.fc1Weight = LoadWeigth(model.context, pickleLoader, tensor);
					model.fc1Weight.Name = "fc1.weight";
				}
				else if (tensor.Name == "fc1.bias")
				{
					model.fc1Bias = LoadWeigth(model.context, pickleLoader, tensor);
					model.fc1Bias.Name = "fc1.bias";
				}
			}
			return model;
		}

		private static SafeGGmlTensor LoadWeigth(SafeGGmlContext context, PickleLoader pickleLoader, Tensor commonTensor)
		{
			SafeGGmlTensor tensor = context.NewTensor(commonTensor.Type, commonTensor.Shape.ToArray());
			byte[] bytes = pickleLoader.ReadByteFromFile(commonTensor);
			tensor.SetData(bytes);
			return tensor;
		}

		private static void BuildGraph(Model model)
		{
			model.graph = model.context.NewGraph();

			SafeGGmlTensor re = model.context.MulMat(model.context.Reshape2d(model.fc1Weight, model.fc1Weight.Shape[1], model.fc1Weight.Shape[0]), model.input);
			re = model.context.Add(re, model.fc1Bias);
			re = model.context.Relu(re);
			re = model.context.MulMat(model.context.Reshape2d(model.fc2Weight, model.fc2Weight.Shape[1], model.fc2Weight.Shape[0]), re);
			re = model.context.Add(re, model.fc2Bias);
			re = model.context.SoftMax(re);
			re.Name = "probs";

			model.graph.BuildForwardExpend(re);
		}

		// compute with backend
		private static SafeGGmlTensor Compute(Model model, SafeGGmlGraphAllocr allocr)
		{
			// allocate tensors
			model.graph.GraphAllocate(allocr);

			model.graph.BackendCompute(model.backend);

			// in this case, the output tensor is the last one in the graph
			return model.graph.Nodes[model.graph.NodeCount - 1];
		}



	}
}
