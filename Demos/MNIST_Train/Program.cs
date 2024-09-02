using GGMLSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static GGMLSharp.Structs;

namespace MNIST_Train
{
	internal class Program
	{
		static void Main(string[] args)
		{
			// Loop count
			int lp = 10;

			// Step count
			int sp = 300;

			mnist_data[] datas = LoadData(@".\Assets\t10k-images.idx3-ubyte", @".\Assets\t10k-labels-idx1-ubyte");

			SafeGGmlContext context = new SafeGGmlContext();

			SafeGGmlTensor input = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 28 * 28);
			SafeGGmlTensor fc1Weight = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 784, 500);
			SafeGGmlTensor fc1Bias = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 500);
			SafeGGmlTensor fc2Weight = context.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 500, 10);
			SafeGGmlTensor fc2Bias = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 10);

			fc1Weight.Name = "fc1Weight";
			fc1Bias.Name = "fc1Bias";
			fc2Weight.Name = "fc2Weight";
			fc2Bias.Name = "fc2Bias";

			context.SetParam(fc1Weight);
			context.SetParam(fc1Bias);
			context.SetParam(fc2Weight);
			context.SetParam(fc2Bias);

			SafeGGmlTensor re = context.Linear(input, fc1Weight, fc1Bias);
			re = context.Relu(re);
			re = context.Linear(re, fc2Weight, fc2Bias);
			SafeGGmlTensor probs = context.SoftMax(re);
			SafeGGmlTensor label = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 10);
			SafeGGmlTensor loss = context.CrossEntropyLoss(probs, label);

			SafeGGmlGraph gf = context.CustomNewGraph();
			gf.BuildForwardExpend(loss);

			float rnd_max = 0.1f;
			float rnd_min = -0.1f;
			fc1Weight.SetRandomTensorInFloat(rnd_max, rnd_min);
			fc1Bias.SetRandomTensorInFloat(rnd_max, rnd_min);
			fc2Weight.SetRandomTensorInFloat(rnd_max, rnd_min);
			fc2Bias.SetRandomTensorInFloat(rnd_max, rnd_min);

			for (int loop = 0; loop < lp; loop++)
			{
				for (int step = 0; step < sp; step++)
				{
					input.SetData(datas[step].data);
					float[] labels = new float[10];
					labels[datas[step].label] = 1;
					label.SetData(labels);
					gf.Reset();
					gf.ComputeWithGGmlContext(context, 1);

					float ls0 = loss.GetFloat();
					List<float> probs_data = probs.GetDataInFloats().ToList();
					int index = probs_data.IndexOf(probs_data.Max());

					OptimizerParameters opt_params = SafeGGmlContext.GetDefaultOptimizerParams(Structs. OptimizerType.ADAM);
					opt_params.PrintBackwardGraph = Convert.ToByte(false);
					opt_params.PrintForwarGraph = Convert.ToByte(false);
				
					Structs.OptimizationResult result = SafeGGmlContext.OptimizerWithDefaultGGmlContext(opt_params, loss);
					Console.WriteLine("loop: {0,3}, setp {1,3} label: {2}, prediction: {3}, match:{4}, loss: {5},", loop, step, datas[step].label, index, datas[step].label == index, ls0);
				}
			}

			Console.WriteLine("Training finished, saving model to mnist_train.gguf");
			SafeGGufContext gguf = SafeGGufContext.Initialize();
			gguf.AddTensor(fc1Weight);
			gguf.AddTensor(fc1Bias);
			gguf.AddTensor(fc2Weight);
			gguf.AddTensor(fc2Bias);
			gguf.WriteToFile("mnist_train.gguf", false);
			gguf.Free();

			Console.WriteLine("Model saved, testing model......");

			TestModel();

			Console.ReadKey();
		}

		class mnist_data
		{
			public float[] data;
			public int label;
		}

		private static mnist_data[] LoadData(string imagePath, string labelPath)
		{
			byte[] imageBytes = File.ReadAllBytes(imagePath);
			byte[] labelBytes = File.ReadAllBytes(labelPath);
			int count = (imageBytes.Length - 16) / (28 * 28);
			mnist_data[] datas = new mnist_data[count];
			for (int i = 0; i < count; i++)
			{
				datas[i] = new mnist_data();
				datas[i].data = new float[28 * 28];
				datas[i].label = labelBytes[8 + i];
				for (int j = 0; j < 28 * 28; j++)
				{
					datas[i].data[j] = imageBytes[16 + i * 28 * 28 + j] / 255.0f;
				}
			}
			return datas;
		}

		private static void DrawImage(mnist_data data)
		{
			Console.WriteLine($"The value is:{data.label}");
			for (int i = 0; i < 28; i++)
			{
				for (int j = 0; j < 28; j++)
				{
					Console.Write(data.data[i * 28 + j] > 0.5 ? " " : "*");
				}
				Console.WriteLine();
			}
		}

		private static void TestModel()
		{
			SafeGGmlContext ctx0 = new SafeGGmlContext();

			SafeGGufContext gguf = SafeGGufContext.InitFromFile("mnist_train.gguf", ctx0, false);
			SafeGGmlTensor fc1Weight = ctx0.GetTensor("fc1Weight");
			SafeGGmlTensor fc1Bias = ctx0.GetTensor("fc1Bias");
			SafeGGmlTensor fc2Weight = ctx0.GetTensor("fc2Weight");
			SafeGGmlTensor fc2Bias = ctx0.GetTensor("fc2Bias");


			SafeGGmlContext context = new SafeGGmlContext();
			SafeGGmlTensor input = context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 28 * 28);
			SafeGGmlTensor re = context.MulMat(fc1Weight, input);
			re = context.Add(re, fc1Bias);
			re = context.Relu(re);
			re = context.MulMat(fc2Weight, re);
			re = context.Add(re, fc2Bias);
			SafeGGmlTensor probs = context.SoftMax(re);
			SafeGGmlGraph gf = context.CustomNewGraph();
			gf.BuildForwardExpend(probs);

			mnist_data[] datas = LoadData(@".\Assets\t10k-images.idx3-ubyte", @".\Assets\t10k-labels-idx1-ubyte");

			mnist_data data = datas[5008];
			input.SetData(data.data);
			gf.ComputeWithGGmlContext(context, 1);

			List<float> probs_data = probs.GetDataInFloats().ToList();
			int index = probs_data.IndexOf(probs_data.Max());
			Console.WriteLine("label: {0}, prediction: {1}, match:{2}", data.label, index, data.label == index);

		}
	}
}
