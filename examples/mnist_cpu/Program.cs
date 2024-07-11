using GGMLSharp;
using ModelLoader;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
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
			SafeGGmlGraph graph = model.context.NewGraph();

			SafeGGmlTensor input = model.context.NewTensor1d(GGmlType.GGML_TYPE_F32, 28 * 28);
			input.SetData(digit);
			input.Name = "input";

			SafeGGmlTensor re = model.context.MulMat(model.fc1Weight, input);
			re = model.context.Add(re, model.fc1Bias);
			re = model.context.Relu(re);
			re = model.context.MulMat(model.fc2Weight, re);
			re = model.context.Add(re, model.fc2Bias);
			SafeGGmlTensor probs = model.context.SoftMax(re);
			probs.Name = "probs";

			graph.BuildForwardExpend(probs);

			graph.ComputeWithGGmlContext(model.context, n_threads);

			List<float> probsList = probs.GetDataInFloats().ToList(); ;
			int prediction = probsList.IndexOf(probsList.Max());
			model.context.Free();

			return prediction;
		}


		public class Model
		{
			public SafeGGmlTensor fc2Weight;
			public SafeGGmlTensor fc2Bias;
			public SafeGGmlTensor fc1Weight;
			public SafeGGmlTensor fc1Bias;
			public SafeGGmlContext context;
		}

		public static Model LoadModel(string path)
		{
			SafeGGmlContext context = new SafeGGmlContext();
			ModelLoader.PickleLoader pickleLoader = new ModelLoader.PickleLoader();
			List<ModelLoader.Tensor> tensors = pickleLoader.ReadTensorsInfoFromFile(path);
			Model model = new Model();
			model.context = context;
			foreach (var tensor in tensors)
			{
				if (tensor.Name == "fc2.weight")
				{
					model.fc2Weight = LoadWeigth(context, pickleLoader, tensor);
				}
				else if (tensor.Name == "fc2.bias")
				{
					model.fc2Bias = LoadWeigth(context, pickleLoader, tensor);
				}
				else if (tensor.Name == "fc1.weight")
				{
					model.fc1Weight = LoadWeigth(context, pickleLoader, tensor);
				}
				else if (tensor.Name == "fc1.bias")
				{
					model.fc1Bias = LoadWeigth(context, pickleLoader, tensor);
				}
			}
			return model;
		}

		private static SafeGGmlTensor LoadWeigth(SafeGGmlContext context, PickleLoader pickleLoader, ModelLoader.Tensor commonTensor)
		{
			SafeGGmlTensor tensor = context.NewTensor(commonTensor.Type, commonTensor.Shape.ToArray());
			byte[] bytes = pickleLoader.ReadByteFromFile(commonTensor);
			tensor.SetData(bytes);
			if (commonTensor.Name.Contains("weight"))
			{
				tensor = context.Transpose(tensor);
				Marshal.Copy(tensor.Data, bytes, 0, bytes.Length);
				tensor = context.NewTensor2d(commonTensor.Type, commonTensor.Shape[1], commonTensor.Shape[0]);
				tensor.SetData(bytes);
			}
			tensor.Name = commonTensor.Name;
			return tensor;
		}
	}
}
