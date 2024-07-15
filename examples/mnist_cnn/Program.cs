using GGMLSharp;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;


namespace MNIST_CNN
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("MNIST-CNN Demo");
			Console.WriteLine($"Has Cuda: {Common.HasCuda}");

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

			MNISTmodel model = MNISTmodelLoad(@".\Assets\mnist-cnn-model.gguf");
			int prediction = Eval(model, 1, digit, string.Empty);
			Console.WriteLine("Prediction: {0}", prediction);
			Console.ReadKey();
		}
		private class MNISTmodel
		{
			public SafeGGmlTensor conv2d1Kernel;
			public SafeGGmlTensor conv2d1Bias;
			public SafeGGmlTensor conv2d2Kernel;
			public SafeGGmlTensor conv2d2Bias;
			public SafeGGmlTensor denseWeight;
			public SafeGGmlTensor denseBias;
			public SafeGGmlContext ctx = new SafeGGmlContext();
		};

		private static MNISTmodel MNISTmodelLoad(string fname)
		{
			MNISTmodel model = new MNISTmodel();
			SafeGGufContext ctx = SafeGGufContext.InitFromFile(fname, model.ctx, false);
			if (!ctx.IsHeaderMagicMatch)
			{
				throw new FileLoadException("gguf_init_from_file() failed");
			}
			model.conv2d1Kernel = model.ctx.GetTensor("kernel1");
			model.conv2d1Bias = model.ctx.GetTensor("bias1");
			model.conv2d2Kernel = model.ctx.GetTensor("kernel2");
			model.conv2d2Bias = model.ctx.GetTensor("bias2");
			model.denseWeight = model.ctx.GetTensor("dense_w");
			model.denseBias = model.ctx.GetTensor("dense_b");
			return model;
		}

		private static int Eval(MNISTmodel model, int threads, float[] digit, string graphName)
		{
			SafeGGmlContext context = new SafeGGmlContext();
			SafeGGmlGraph graph = context.NewGraph();

			SafeGGmlTensor input = context.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, 28, 28, 1, 1);
			input.SetData(digit);
			input.Name = "input";

			SafeGGmlTensor cur = context.Conv2d(input, model.conv2d1Kernel, model.conv2d1Bias);
			cur = context.Relu(cur);
			// Output shape after Conv2D: (26 26 32 1)
			cur = context.Pool2d(cur);
			// Output shape after MaxPooling2D: (13 13 32 1)
			cur = context.Conv2d(cur,model.conv2d2Kernel , model.conv2d2Bias);

			cur = context.Relu(cur);
			// Output shape after Conv2D: (11 11 64 1)
			cur = context.Pool2d(cur);
			// Output shape after MaxPooling2D: (5 5 64 1)
			cur = context.Permute(cur, 1, 2, 0, 3);
			cur = context.Cont(cur);
			// Output shape after permute: (64 5 5 1)
			cur = context.Reshape2d(cur, 1600, 1);
			// Final Dense layer
			cur = context.Linear(cur, model.denseWeight, model.denseBias);

			SafeGGmlTensor probs = context.SoftMax(cur);
			probs.Name = "probs";

			graph.BuildForwardExpend(probs);

			Stopwatch stopwatch = Stopwatch.StartNew();

			graph.ComputeWithGGmlContext(context, threads);

			stopwatch.Stop();
			Console.WriteLine("compute Time: {0} ticks.", stopwatch.ElapsedTicks);

			//ggml_graph_print(&graph);
			//Native.ggml_graph_dump_dot(graph, null, "mnist-cnn.dot");

			if (!string.IsNullOrEmpty(graphName))
			{
				// export the compute graph for later use
				// see the "mnist-cpu" example
				graph.Export(graphName);
				Console.WriteLine("exported compute graph to {0}\n", graphName);
			}
			float[] probs_list = probs.GetDataInFloats();
			int prediction = probs_list.ToList().IndexOf(probs_list.Max());
			model.ctx.Free();
			context.Free();
			return prediction;
		}
	}
}
