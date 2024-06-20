using GGMLSharp;
using System.Diagnostics;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace mnist_cnn
{
	internal unsafe class Program
	{

		// A simple Demo for MNIST-CNN
		static void Main(string[] args)
		{
			Console.WriteLine("MNIST-CNN Demo");
			Console.WriteLine($"Has Cuda: {Native.ggml_cpu_has_cuda()}");

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

			mnist_model model = mnist_model_load(@".\Assets\mnist-cnn-model.gguf");
			Stopwatch stopwatch = Stopwatch.StartNew();

			int prediction = mnist_eval(model, 1, digit, string.Empty);

			Console.WriteLine("Prediction: {0}", prediction);
			Console.ReadKey();
		}


		private struct mnist_model
		{
			public ggml_tensor* conv2d_1_kernel;
			public ggml_tensor* conv2d_1_bias;
			public ggml_tensor* conv2d_2_kernel;
			public ggml_tensor* conv2d_2_bias;
			public ggml_tensor* dense_weight;
			public ggml_tensor* dense_bias;
			public ggml_context* ctx;
		};

		private static mnist_model mnist_model_load(string fname)
		{
			mnist_model model = new mnist_model();
			gguf_init_params @params = new gguf_init_params
			{
				ctx = &model.ctx,
				no_alloc = false,
			};
			gguf_context* ctx = Native.gguf_init_from_file(fname, @params);
			if (ctx == null)
			{
				throw new FileLoadException("gguf_init_from_file() failed");
			}
			Native.gguf_free(ctx);
			model.conv2d_1_kernel = Native.ggml_get_tensor(model.ctx, "kernel1");
			model.conv2d_1_bias = Native.ggml_get_tensor(model.ctx, "bias1");
			model.conv2d_2_kernel = Native.ggml_get_tensor(model.ctx, "kernel2");
			model.conv2d_2_bias = Native.ggml_get_tensor(model.ctx, "bias2");
			model.dense_weight = Native.ggml_get_tensor(model.ctx, "dense_w");
			model.dense_bias = Native.ggml_get_tensor(model.ctx, "dense_b");
			return model;
		}

		private static int mnist_eval(mnist_model model, int n_threads, float[] digit, string fname_cgraph)
		{
			ulong buf_size = 1024 * 1024; // Get 1M mem size
			ggml_init_params @params = new ggml_init_params
			{
				mem_buffer = IntPtr.Zero,
				mem_size = buf_size,
				no_alloc = false,
			};

			ggml_context* ctx0 = Native.ggml_init(@params);
			ggml_cgraph* gf = Native.ggml_new_graph(ctx0);

			ggml_tensor* input = Native.ggml_new_tensor_4d(ctx0, ggml_type.GGML_TYPE_F32, 28, 28, 1, 1);
			Marshal.Copy(digit, 0, input->data, digit.Length);

			Native.ggml_set_name(input, "input");
			ggml_tensor* cur = Native.ggml_conv_2d(ctx0, model.conv2d_1_kernel, input, 1, 1, 0, 0, 1, 1);
			cur = Native.ggml_add(ctx0, cur, model.conv2d_1_bias);

			cur = Native.ggml_relu(ctx0, cur);
			// Output shape after Conv2D: (26 26 32 1)
			cur = Native.ggml_pool_2d(ctx0, cur, ggml_op_pool.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
			// Output shape after MaxPooling2D: (13 13 32 1)
			cur = Native.ggml_conv_2d(ctx0, model.conv2d_2_kernel, cur, 1, 1, 0, 0, 1, 1);
			cur = Native.ggml_add(ctx0, cur, model.conv2d_2_bias);
			cur = Native.ggml_relu(ctx0, cur);
			// Output shape after Conv2D: (11 11 64 1)
			cur = Native.ggml_pool_2d(ctx0, cur, ggml_op_pool.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
			// Output shape after MaxPooling2D: (5 5 64 1)
			cur = Native.ggml_permute(ctx0, cur, 1, 2, 0, 3);
			cur = Native.ggml_cont(ctx0, cur);
			// Output shape after permute: (64 5 5 1)
			cur = Native.ggml_reshape_2d(ctx0, cur, 1600, 1);
			// Final Dense layer
			cur = Native.ggml_mul_mat(ctx0, model.dense_weight, cur);
			cur = Native.ggml_add(ctx0, cur, model.dense_bias);
			ggml_tensor* probs = Native.ggml_soft_max(ctx0, cur);
			Native.ggml_set_name(probs, "probs");

			Native.ggml_build_forward_expand(gf, probs);

			Stopwatch stopwatch = Stopwatch.StartNew();

			Native.ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

			stopwatch.Stop();
			Console.WriteLine("compute Time: {0} ticks.", stopwatch.ElapsedTicks);

			//ggml_graph_print(&gf);
			//Native.ggml_graph_dump_dot(gf, null, "mnist-cnn.dot");

			if (!string.IsNullOrEmpty(fname_cgraph))
			{
				// export the compute graph for later use
				// see the "mnist-cpu" example
				Native.ggml_graph_export(gf, fname_cgraph);
				Console.WriteLine("exported compute graph to {0}\n", fname_cgraph);
			}

			float* probs_data = Native.ggml_get_data_f32(probs);

			List<float> probs_list = new List<float>();
			for (int i = 0; i < 10; i++)
			{
				probs_list.Add(probs_data[i]);
			}
			int prediction = probs_list.IndexOf(probs_list.Max());
			Native.ggml_free(ctx0);

			return prediction;
		}


	}
}
