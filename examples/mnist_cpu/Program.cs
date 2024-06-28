using GGMLSharp;
using System.IO;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace mnist_cpu
{
	internal unsafe class Program
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

			int prediction = mnist_eval(model, 1, digit);
			Console.WriteLine("Prediction: {0}", prediction);
		}

		private static int mnist_eval(Model model, int n_threads, float[] digit)
		{
			ggml_cgraph* gf = Native.ggml_new_graph(model.context);

			ggml_tensor* input = Native.ggml_new_tensor_1d(model.context, ggml_type.GGML_TYPE_F32, 28 * 28);
			Marshal.Copy(digit, 0, input->data, digit.Length);
			Native.ggml_set_name(input, "input");

			ggml_tensor* re = Native.ggml_mul_mat(model.context, model.fc1_weight, input);
			re = Native.ggml_add(model.context, re, model.fc1_bias);
			re = Native.ggml_relu(model.context, re);
			re = Native.ggml_mul_mat(model.context, model.fc2_weight, re);
			re = Native.ggml_add(model.context, re, model.fc2_bias);
			ggml_tensor* probs = Native.ggml_soft_max(model.context, re);
			Native.ggml_set_name(probs, "probs");

			Native.ggml_build_forward_expand(gf, probs);

			Native.ggml_graph_compute_with_ctx(model.context, gf, n_threads);

			float* probs_data = Native.ggml_get_data_f32(probs);

			List<float> probs_list = new List<float>();
			for (int i = 0; i < 10; i++)
			{
				probs_list.Add(probs_data[i]);
			}
			int prediction = probs_list.IndexOf(probs_list.Max());
			Native.ggml_free(model.context);

			return prediction;
		}


		public class Model
		{
			public ggml_tensor* fc2_weight;
			public ggml_tensor* fc2_bias;
			public ggml_tensor* fc1_weight;
			public ggml_tensor* fc1_bias;
			public ggml_context* context;
		}

		public static Model LoadModel(string path)
		{
			ggml_init_params @params = new ggml_init_params
			{
				mem_buffer = IntPtr.Zero,
				mem_size = 512 * 1024 * 1024,
				no_alloc = false,
			};
			ggml_context* context = Native.ggml_init(@params);
			ModelLoader.PickleLoader pickleLoader = new ModelLoader.PickleLoader();
			List<ModelLoader.Tensor> tensors = pickleLoader.ReadTensorsInfoFromFile(path);
			Model model = new Model();
			model.context = context;
			foreach (var tensor in tensors)
			{
				if (tensor.Name == "fc2.weight")
				{
					model.fc2_weight = LoadWeigth(context, tensor);
				}
				else if (tensor.Name == "fc2.bias")
				{
					model.fc2_bias = LoadWeigth(context, tensor);
				}
				else if (tensor.Name == "fc1.weight")
				{
					model.fc1_weight = LoadWeigth(context, tensor);
				}
				else if (tensor.Name == "fc1.bias")
				{
					model.fc1_bias = LoadWeigth(context, tensor);
				}
			}
			return model;
		}

		private static ggml_tensor* LoadWeigth(ggml_context* context, ModelLoader.Tensor commonTensor)
		{
			ggml_tensor* tensor = Native.ggml_new_tensor(context, commonTensor.Type, commonTensor.Shape.Count, commonTensor.Shape.ToArray());
			byte[] bytes = new ModelLoader.PickleLoader().ReadByteFromFile(commonTensor);
			Marshal.Copy(bytes, 0, tensor->data, bytes.Length);
			if (commonTensor.Name.Contains("weight"))
			{
				tensor = Native.ggml_transpose(context, tensor);
				Marshal.Copy(tensor->data, bytes, 0, bytes.Length);
				tensor = Native.ggml_new_tensor_2d(context, commonTensor.Type, commonTensor.Shape[1], commonTensor.Shape[0]);
				Marshal.Copy(bytes, 0, tensor->data, bytes.Length);
			}
			Native.ggml_set_name(tensor, commonTensor.Name);
			return tensor;
		}

	}
}
