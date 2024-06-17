using GGMLSharp;
using static GGMLSharp.Structs;

namespace mnist_train
{
	internal unsafe class Program
	{
		static void Main(string[] args)
		{
			// Loop count
			int lp = 20;

			// Step count
			int sp = 1000;

			mnist_data[] datas = LoadData(@".\Assets\t10k-images.idx3-ubyte", @".\Assets\t10k-labels-idx1-ubyte");
			ggml_init_params init_params = new ggml_init_params
			{
				mem_size = 10 * 1024 * 1024,
				mem_buffer = IntPtr.Zero,
				no_alloc = false,
			};

			ggml_context* context = Native.ggml_init(init_params);

			ggml_tensor* input = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 28 * 28);
			ggml_tensor* fc1_weight = Native.ggml_new_tensor_2d(context, ggml_type.GGML_TYPE_F32, 784, 500);
			ggml_tensor* fc1_bias = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 500);
			ggml_tensor* fc2_weight = Native.ggml_new_tensor_2d(context, ggml_type.GGML_TYPE_F32, 500, 10);
			ggml_tensor* fc2_bias = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 10);

			Native.ggml_set_param(context, fc1_weight);
			Native.ggml_set_param(context, fc1_bias);
			Native.ggml_set_param(context, fc2_weight);
			Native.ggml_set_param(context, fc2_bias);

			ggml_tensor* re = Native.ggml_mul_mat(context, fc1_weight, input);
			re = Native.ggml_add(context, re, fc1_bias);
			re = Native.ggml_relu(context, re);
			re = Native.ggml_mul_mat(context, fc2_weight, re);
			re = Native.ggml_add(context, re, fc2_bias);
			ggml_tensor* probs = Native.ggml_soft_max(context, re);
			ggml_tensor* label = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 10);
			ggml_tensor* loss = Native.ggml_cross_entropy_loss(context, probs, label);

			Native.ggml_set_name(fc1_weight, "fc1_weight");
			Native.ggml_set_name(fc1_bias, "fc1_bias");
			Native.ggml_set_name(fc2_weight, "fc2_weight");
			Native.ggml_set_name(fc2_bias, "fc2_bias");

			ggml_cgraph* gf = Native.ggml_new_graph_custom(context, GGML_DEFAULT_GRAPH_SIZE, true);
			Native.ggml_build_forward_expand(gf, loss);

			float rnd_max = 0.1f;
			float rnd_min = -0.1f;

			SetRandomValues(fc1_weight, rnd_max, rnd_min);
			SetRandomValues(fc1_bias, rnd_max, rnd_min);
			SetRandomValues(fc2_weight, rnd_max, rnd_min);
			SetRandomValues(fc2_bias, rnd_max, rnd_min);

			for (int loop = 0; loop < lp; loop++)
			{
				for (int step = 0; step < sp; step++)
				{
					SetInputValues(input, datas[step].data);

					for (int i = 0; i < 10; i++)
					{
						Native.ggml_set_f32_1d(label, i, i == datas[step].label ? 1 : 0);
					}
					Native.ggml_graph_reset(gf);
					Native.ggml_graph_compute_with_ctx(context, gf, 1);

					float ls0 = Native.ggml_get_f32_1d(loss, 0);
					float* probs_data = Native.ggml_get_data_f32(probs);

					int index = 0;
					float temp_max = 0;
					for (int i = 0; i < 10; i++)
					{
						if (probs_data[i] > temp_max)
						{
							temp_max = probs_data[i];
							index = i;
						}
					}

					ggml_opt_params opt_params = Native.ggml_opt_default_params(ggml_opt_type.GGML_OPT_TYPE_ADAM);
					opt_params.print_backward_graph = false;
					opt_params.print_forward_graph = false;
					ggml_opt_result result = Native.ggml_opt(null, opt_params, loss);
					Console.WriteLine("loop: {0,3}, setp {1,3} label: {2}, prediction: {3}, match:{4}, loss: {5},", loop, step, datas[step].label, index, datas[step].label == index, ls0);
				}
			}

			Console.WriteLine("Training finished, saving model to mnist_train.gguf");
			gguf_context* gguf = Native.gguf_init_empty();
			Native.gguf_add_tensor(gguf, fc1_weight);
			Native.gguf_add_tensor(gguf, fc1_bias);
			Native.gguf_add_tensor(gguf, fc2_weight);
			Native.gguf_add_tensor(gguf, fc2_bias);
			Native.gguf_write_to_file(gguf, "mnist_train.gguf", false);
			Native.gguf_free(gguf);

			Console.WriteLine("Model saved, testing model......");

			TestModel();

		}

		private static void SetRandomValues(ggml_tensor* tensor, float max, float min)
		{
			for (int i = 0; i < tensor->ne[0]; i++)
			{
				for (int j = 0; j < tensor->ne[1]; j++)
				{
					for (int k = 0; k < tensor->ne[2]; k++)
					{
						for (int l = 0; l < tensor->ne[3]; l++)
						{
							Native.ggml_set_f32_nd(tensor, i, j, k, l, new Random((int)DateTime.Now.Ticks).NextSingle() * (max - min) + min);
						}
					}
				}
			}
		}

		private static void SetInputValues(ggml_tensor* tensor, float[] values)
		{
			for (int i = 0; i < values.Length; i++)
			{
				Native.ggml_set_f32_1d(tensor, i, values[i]);
			}

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
			ggml_init_params init_params = new ggml_init_params
			{
				mem_size = 10 * 1024 * 1024,
				mem_buffer = IntPtr.Zero,
				no_alloc = false,
			};
			ggml_context* ctx0 = Native.ggml_init(init_params);

			gguf_init_params gguf_init_params = new gguf_init_params
			{
				ctx = &ctx0,
				no_alloc = false,
			};
			gguf_context* gguf = Native.gguf_init_from_file("mnist_train.gguf", gguf_init_params);
			ggml_tensor* fc1_weight = Native.ggml_get_tensor(ctx0, "fc1_weight");
			ggml_tensor* fc1_bias = Native.ggml_get_tensor(ctx0, "fc1_bias");
			ggml_tensor* fc2_weight = Native.ggml_get_tensor(ctx0, "fc2_weight");
			ggml_tensor* fc2_bias = Native.ggml_get_tensor(ctx0, "fc2_bias");


			ggml_context* context = Native.ggml_init(init_params);
			ggml_tensor* input = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 28 * 28);
			ggml_tensor* re = Native.ggml_mul_mat(context, fc1_weight, input);
			re = Native.ggml_add(context, re, fc1_bias);
			re = Native.ggml_relu(context, re);
			re = Native.ggml_mul_mat(context, fc2_weight, re);
			re = Native.ggml_add(context, re, fc2_bias);
			ggml_tensor* probs = Native.ggml_soft_max(context, re);
			ggml_cgraph* gf = Native.ggml_new_graph_custom(context, GGML_DEFAULT_GRAPH_SIZE, true);
			Native.ggml_build_forward_expand(gf, probs);

			mnist_data[] datas = LoadData(@".\Assets\t10k-images.idx3-ubyte", @".\Assets\t10k-labels-idx1-ubyte");

			mnist_data data = datas[5006];
			SetInputValues(input, data.data);
			Native.ggml_graph_compute_with_ctx(context, gf, 1);
			float* probs_data = Native.ggml_get_data_f32(probs);
			int index = 0;
			float temp_max = 0;
			for (int j = 0; j < 10; j++)
			{
				if (probs_data[j] > temp_max)
				{
					temp_max = probs_data[j];
					index = j;
				}
			}
			Console.WriteLine("label: {0}, prediction: {1}, match:{2}", data.label, index, data.label == index);

		}
	}
}

