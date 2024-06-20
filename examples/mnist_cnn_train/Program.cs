using GGMLSharp;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace mnist_cnn_train
{
	internal unsafe class Program
	{
		static void Main(string[] args)
		{
			int lp = 20;
			int sp = 1000;

			ggml_init_params init_params = new ggml_init_params
			{
				mem_size = 512u * 1024 * 1024,
				mem_buffer = IntPtr.Zero,
				no_alloc = false,
			};

			ggml_context* context = Native.ggml_init(init_params);
			ggml_tensor* input = Native.ggml_new_tensor_2d(context, ggml_type.GGML_TYPE_F32, 28, 28);
			ggml_tensor* conv2d_1_kernel = Native.ggml_new_tensor_4d(context, ggml_type.GGML_TYPE_F16, 3, 3, 1, 32);
			ggml_tensor* conv2d_1_bias = Native.ggml_new_tensor_3d(context, ggml_type.GGML_TYPE_F32, 26, 26, 32);
			ggml_tensor* conv2d_2_kernel = Native.ggml_new_tensor_4d(context, ggml_type.GGML_TYPE_F16, 3, 3, 32, 64);
			ggml_tensor* conv2d_2_bias = Native.ggml_new_tensor_3d(context, ggml_type.GGML_TYPE_F32, 11, 11, 64);
			ggml_tensor* dense_weight = Native.ggml_new_tensor_2d(context, ggml_type.GGML_TYPE_F32, 1600, 10);
			ggml_tensor* dense_bias = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 10);

			Native.ggml_set_param(context, conv2d_1_kernel);
			Native.ggml_set_param(context, conv2d_1_bias);
			Native.ggml_set_param(context, conv2d_2_kernel);
			Native.ggml_set_param(context, conv2d_2_bias);
			Native.ggml_set_param(context, dense_weight);
			Native.ggml_set_param(context, dense_bias);

			ggml_tensor* re = Native.ggml_conv_2d(context, conv2d_1_kernel, input, 1, 1, 0, 0, 1, 1);
			re = Native.ggml_add(context, re, conv2d_1_bias);
			re = Native.ggml_relu(context, re);
			// Output shape after Conv2D: (26 26 32 1)
			re = Native.ggml_pool_2d(context, re, ggml_op_pool.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
			// Output shape after MaxPooling2D: (13 13 32 1)
			re = Native.ggml_conv_2d(context, conv2d_2_kernel, re, 1, 1, 0, 0, 1, 1);
			re = Native.ggml_add(context, re, conv2d_2_bias);
			re = Native.ggml_relu(context, re);
			// Output shape after Conv2D: (11 11 64 1)
			re = Native.ggml_pool_2d(context, re, ggml_op_pool.GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
			// Output shape after MaxPooling2D: (5 5 64 1)
			re = Native.ggml_permute(context, re, 1, 2, 0, 3);
			re = Native.ggml_cont(context, re);
			// Output shape after permute: (64 5 5 1)
			re = Native.ggml_reshape_2d(context, re, 1600, 1);
			// Final Dense layer
			re = Native.ggml_mul_mat(context, dense_weight, re);
			re = Native.ggml_add(context, re, dense_bias);
			ggml_tensor* probs = Native.ggml_soft_max(context, re);
			ggml_tensor* label = Native.ggml_new_tensor_1d(context, ggml_type.GGML_TYPE_F32, 10);
			ggml_tensor* loss = Native.ggml_cross_entropy_loss(context, probs, label);

			ggml_cgraph* gf = Native.ggml_new_graph_custom(context, GGML_DEFAULT_GRAPH_SIZE, true);
			Native.ggml_build_forward_expand(gf, loss);
			ggml_cgraph* gb = Native.ggml_graph_dup(context, gf);
			Native.ggml_build_backward_expand(context, gf, gb, true);

			float rnd_max = 0.1f;
			float rnd_min = -0.1f;

			SetRandomValues(conv2d_1_kernel, rnd_max, rnd_min);
			SetRandomValues(conv2d_1_bias, rnd_max, rnd_min);
			SetRandomValues(conv2d_2_kernel, rnd_max, rnd_min);
			SetRandomValues(conv2d_2_bias, rnd_max, rnd_min);
			SetRandomValues(dense_weight, rnd_max, rnd_min);
			SetRandomValues(dense_bias, rnd_max, rnd_min);

			mnist_data[] datas = LoadData(@".\Assets\t10k-images.idx3-ubyte", @".\Assets\t10k-labels-idx1-ubyte");
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
					Native.ggml_graph_reset(gb);
					Native.ggml_graph_compute_with_ctx(context, gb, 1);

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
					Console.WriteLine("loop: {0,3}, setp {1,3} label: {2}, prediction: {3}, match:{4}, loss: {5},", loop, step, datas[step].label, index, datas[step].label == index, ls0);

					//UpdateWeight(conv2d_1_kernel, conv2d_1_kernel->grad);
					//UpdateWeight(conv2d_1_bias, conv2d_1_bias->grad);
					//UpdateWeight(conv2d_2_kernel, conv2d_2_kernel->grad);
					//UpdateWeight(conv2d_2_bias, conv2d_2_bias->grad);
					//UpdateWeight(dense_weight, dense_weight->grad);
					//UpdateWeight(dense_bias, dense_bias->grad);

					ggml_opt_params opt_params = Native.ggml_opt_default_params(ggml_opt_type.GGML_OPT_TYPE_ADAM);
					opt_params.print_backward_graph = false;
					opt_params.print_forward_graph = false;
					ggml_opt_result result = Native.ggml_opt(null, opt_params, loss);

				}
			}

			Native.ggml_set_name(conv2d_1_kernel, "kernel1");
			Native.ggml_set_name(conv2d_1_bias, "bias1");
			Native.ggml_set_name(conv2d_2_kernel, "kernel2");
			Native.ggml_set_name(conv2d_2_bias, "bias2");
			Native.ggml_set_name(dense_weight, "dense_w");
			Native.ggml_set_name(dense_bias, "dense_b");

		}

		private static void SetRandomValues(ggml_tensor* tensor, float max, float min)
		{
			int count = (int)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
			float[] floats = new float[count];
			for (int i = 0; i < count; i++)
			{
				floats[i] = new Random((int)DateTime.Now.Ticks).NextSingle() * (max - min) + min;
			}
			List<byte> bytes = new List<byte>();
			foreach (var f in floats)
			{
				if (tensor->type == ggml_type.GGML_TYPE_F32)
				{
					bytes.AddRange(BitConverter.GetBytes(f));
				}
				else if (tensor->type == ggml_type.GGML_TYPE_F16)
				{
					ushort f16 = Native.ggml_fp32_to_fp16(f);
					bytes.AddRange(BitConverter.GetBytes(f16));
				}
				else if (tensor->type == ggml_type.GGML_TYPE_BF16)
				{
					ushort bf16 = Native.ggml_fp32_to_bf16(f);
					bytes.AddRange(BitConverter.GetBytes(bf16));
				}
				else
				{
					throw new Exception("Unsupported type");
				}
			}
			Marshal.Copy(bytes.ToArray(), 0, tensor->data, bytes.Count);
		}

		private static void SetInputValues(ggml_tensor* tensor, float[] values)
		{
			Marshal.Copy(values, 0, tensor->data, values.Length);
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

		public class AdamOptimizer
		{
			private float alpha; // 学习率
			private float beta1; // 一阶矩估计的指数衰减率
			private float beta2; // 二阶矩估计的指数衰减率
			private float epsilon; // 防止除零的常数
			private float[] m; // 一阶矩估计
			private float[] v; // 二阶矩估计
			private int t; // 时间步

			public AdamOptimizer(float alpha = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
			{
				this.alpha = alpha;
				this.beta1 = beta1;
				this.beta2 = beta2;
				this.epsilon = epsilon;
				this.t = 0;
			}

			public void Initialize(int size)
			{
				m = new float[size];
				v = new float[size];
			}

			public void Update(ref float[] parameters, float[] gradients)
			{
				t++;
				for (int i = 0; i < parameters.Length; i++)
				{
					// 更新一阶矩估计
					m[i] = beta1 * m[i] + (1 - beta1) * gradients[i];
					// 更新二阶矩估计
					v[i] = beta2 * v[i] + (1 - beta2) * gradients[i] * gradients[i];
					// 偏差修正
					float mHat = m[i] / (float)(1 - Math.Pow(beta1, t));
					float vHat = v[i] / (float)(1 - Math.Pow(beta2, t));
					// 更新参数
					parameters[i] -= (float)(alpha * mHat / (Math.Sqrt(vHat) + epsilon));
				}
			}
		}
		private static void UpdateWeight(ggml_tensor* tensor, ggml_tensor* gradients)
		{
			int size = (int)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
			AdamOptimizer optimizer = new AdamOptimizer();
			{
				optimizer.Initialize(size);
				float[] t = GetFloatsFromTensor(tensor);
				float[] g = GetFloatsFromTensor(gradients);
				optimizer.Update(ref t, g);
				SetTensorValues(tensor, t);
				SetTensorValues(gradients, g);
			}


		}

		private static float[] GetFloatsFromTensor(ggml_tensor* tensor)
		{
			int size = (int)(tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3]);
			byte[] bytes = new byte[size * (int)Native.ggml_type_size(tensor->type)];
			Marshal.Copy(tensor->data, bytes, 0, bytes.Length);
			float[] values = new float[size];
			for (int i = 0; i < size; i++)
			{
				if (tensor->type == ggml_type.GGML_TYPE_F32)
				{
					values[i] = BitConverter.ToSingle(bytes, i * 4);
				}
				else if (tensor->type == ggml_type.GGML_TYPE_F16)
				{
					values[i] = Native.ggml_fp16_to_fp32(BitConverter.ToUInt16(bytes, i * 2));
				}
				else if (tensor->type == ggml_type.GGML_TYPE_BF16)
				{
					values[i] = Native.ggml_bf16_to_fp32(BitConverter.ToUInt16(bytes, i * 2));
				}
				else
				{
					throw new Exception("Unsupported type");
				}
			}
			return values;
		}

		private static void SetTensorValues(ggml_tensor* tensor, float[] values)
		{
			List<byte> bytes = new List<byte>();
			foreach (var f in values)
			{
				if (tensor->type == ggml_type.GGML_TYPE_F32)
				{
					bytes.AddRange(BitConverter.GetBytes(f));
				}
				else if (tensor->type == ggml_type.GGML_TYPE_F16)
				{
					ushort f16 = Native.ggml_fp32_to_fp16(f);
					bytes.AddRange(BitConverter.GetBytes(f16));
				}
				else if (tensor->type == ggml_type.GGML_TYPE_BF16)
				{
					ushort bf16 = Native.ggml_fp32_to_bf16(f);
					bytes.AddRange(BitConverter.GetBytes(bf16));
				}
				else
				{
					throw new Exception("Unsupported type");
				}
			}
			Marshal.Copy(bytes.ToArray(), 0, tensor->data, bytes.Count);
		}

	}
}
