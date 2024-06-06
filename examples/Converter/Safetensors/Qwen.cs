using Converter.Abstractions;
using GGMLSharp;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using static Converter.Safetensors.SafetensorsLoader;
using static GGMLSharp.Structs;

namespace Converter.Safetensors
{
	public unsafe class Qwen : ISafeTensorConverter
    {
		public void Convert(string safetensorsPath, string outputFileName, bool WriteToFileUsingStream = true)
		{
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(safetensorsPath);
			gguf_context* gguf_ctx = Native.gguf_init_empty();
			Native.gguf_set_val_str(gguf_ctx, "general.architecture", "qwen2");
			Native.gguf_set_val_str(gguf_ctx, "general.name", "1_5");
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.block_count", configLoader.num_hidden_layers);
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.context_length", configLoader.max_position_embeddings);
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.embedding_length", configLoader.hidden_size);
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.feed_forward_length", configLoader.intermediate_size);
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.attention.head_count", configLoader.num_attention_heads);
			Native.gguf_set_val_u32(gguf_ctx, "qwen2.attention.head_count_kv", configLoader.num_key_value_heads);
			Native.gguf_set_val_f32(gguf_ctx, "qwen2.rope.freq_base", configLoader.rope_theta);
			Native.gguf_set_val_f32(gguf_ctx, "qwen2.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			Native.gguf_set_val_u32(gguf_ctx, "general.file_type", 1);
			Native.gguf_set_val_str(gguf_ctx, "tokenizer.ggml.model", "gpt2");
			Native.gguf_set_val_str(gguf_ctx, "tokenizer.ggml.pre", "qwen2");
			string[] tokens = configLoader.tokenizer_ggml_tokens.ToArray();

			IntPtr[] datas = new IntPtr[tokens.Length];

			for (int i = 0; i < tokens.Length; i++)
			{
				datas[i] = Marshal.StringToCoTaskMemUTF8(tokens[i]);
			}

			Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.tokens", datas, tokens.Length);
			//Native.gguf_set_arr_str(gguf_ctx, "tokenizer.ggml.tokens", tokens, tokens.Length);
			int[] token_types = configLoader.tokenizer_ggml_token_type.ToArray();
			Native.gguf_set_arr_data(gguf_ctx, "tokenizer.ggml.token_type", gguf_type.GGUF_TYPE_UINT32, Marshal.UnsafeAddrOfPinnedArrayElement(token_types, 0), token_types.Length);
			string[] merges = configLoader.tokenizer_ggml_merges.ToArray();
			IntPtr[] mergesDatas = new IntPtr[merges.Length];
			for (int i = 0; i < merges.Length; i++)
			{
				mergesDatas[i] = Marshal.StringToCoTaskMemUTF8(merges[i]);
			}
			Native.gguf_set_arr_str(gguf_ctx, "tokenizer.ggml.merges", mergesDatas, merges.Length);

			Native.gguf_set_val_u32(gguf_ctx, "tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			Native.gguf_set_val_u32(gguf_ctx, "tokenizer.ggml.padding_token_id", configLoader.pad_token_id);
			Native.gguf_set_val_u32(gguf_ctx, "tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			Native.gguf_set_val_str(gguf_ctx, "tokenizer.chat_template", configLoader.chat_template);

			List<SafetensorsLoader.CommonTensor> safetensors = new List<SafetensorsLoader.CommonTensor>();
			string[] files = Directory.GetFiles(safetensorsPath, "*.safetensors");
			foreach (string file in files)
			{
				safetensors.AddRange(SafetensorsLoader.ReadTensorsInfoFromFile(file));
			}

			Console.WriteLine("Start to load tensors from safetensors file and add to gguf_context.");
			foreach (SafetensorsLoader.CommonTensor tensor in safetensors)
			{
				Console.WriteLine($"tensor name:{tensor.name}");
				ggml_init_params ggml_params = new ggml_init_params
				{
					mem_size = 2 * Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};
				ggml_context* ggml_context = Native.ggml_init(ggml_params);
				long offeset = tensor.offset[0];
				int length = (int)(tensor.offset[1] - tensor.offset[0]);
				byte[] tensorBytes = SafetensorsLoader.ReadByteFromFile(tensor);

				string name = SafetensorsLoader.TensorNameTrans_FromSafetensorsToGguf(tensor.name);
				ggml_type type = tensor.shape.Length == 1 ? ggml_type.GGML_TYPE_F32 : ggml_type.GGML_TYPE_F16;
				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, type, tensor.shape.Length, tensor.shape);
				Native.ggml_set_name(ggml_tensor, name);

				if (name == tensor.name)
				{
					continue;
				}

				if (!WriteToFileUsingStream)
				{
					if (tensor.shape.Length == 1)
					{
						ushort* data_f16 = (ushort*)Marshal.AllocHGlobal(tensorBytes.Length).ToPointer();
						for (int j = 0; j < tensorBytes.Length; j += 2)
						{
							data_f16[j / 2] = (ushort)(tensorBytes[j] | (tensorBytes[j + 1] << 8));
						}
						float* data_fp32 = (float*)Marshal.AllocHGlobal(tensorBytes.Length * 2).ToPointer();

						if (tensor.dtype == ggml_type.GGML_TYPE_F16)
						{
							Native.ggml_fp16_to_fp32_row(data_f16, data_fp32, tensorBytes.Length / 2);
						}
						else if (tensor.dtype == ggml_type.GGML_TYPE_BF16)
						{
							Native.ggml_fp16_to_fp32_row(data_f16, data_fp32, tensorBytes.Length / 2);
						}
						List<byte> output = new List<byte>();
						for (int j = 0; j < tensorBytes.Length / 2; j++)
						{
							output.AddRange(BitConverter.GetBytes(data_fp32[j]));
						}
						tensorBytes = output.ToArray();
						Marshal.FreeHGlobal((IntPtr)data_f16);
						Marshal.FreeHGlobal((IntPtr)data_fp32);
					}
					ggml_tensor->data = Marshal.AllocHGlobal(length);
					Marshal.Copy(tensorBytes, 0, ggml_tensor->data, length);
				}

				if (name == "token_embd.weight" || name == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight"))
				{
					ggml_tensor = Native.ggml_transpose(ggml_context, ggml_tensor);
					Native.ggml_set_name(ggml_tensor, name);
				}

				Native.gguf_add_tensor(gguf_ctx, ggml_tensor);
				Native.ggml_free(ggml_context);
				GC.Collect();
			}

			Console.WriteLine("Add to gguf_context done.");
			Console.WriteLine("Start to write gguf_context to file.");

			if (!WriteToFileUsingStream)
			{
				Console.WriteLine("Write to file using gguf_write_to_file function. Please wait ...");
				Native.gguf_write_to_file(gguf_ctx, "model.gguf", false);
			}
			else
			{
				Console.WriteLine("Write to file using stream.");
				string inputFileName = Path.Combine(safetensorsPath, "model.safetensors");
				//string outputFileName = "model.gguf";

				Native.gguf_write_to_file(gguf_ctx, outputFileName, true);

				byte[] bytes = File.ReadAllBytes(outputFileName);

				long totalSize = 0;
				for (int i = 0; i < (int)gguf_ctx->header.n_tensors; ++i)
				{
					gguf_tensor_info* info = &gguf_ctx->infos[i];
					string name = Marshal.PtrToStringUTF8(info->name.data);
					

					CommonTensor tensor = safetensors.Find(x => SafetensorsLoader.TensorNameTrans_FromSafetensorsToGguf(x.name) == name);
					long size = Math.Max(info->size, (int)gguf_ctx->alignment);
					long _offset = tensor.offset[1] - tensor.offset[0];

					long size_pad = Native.GGML_PAD((int)size, (int)gguf_ctx->alignment);

					byte[] data = ReadByteFromFile(tensor);
					Console.WriteLine($"{name} is doing, bytes to read is {data.Length}, total bytes is {totalSize}");
					string transName = SafetensorsLoader.TensorNameTrans_FromSafetensorsToGguf(tensor.name);

					if (transName == tensor.name)
					{
						continue;
					}
					if (transName == "token_embd.weight" || transName == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight")) //'blk.0.ffn_down.weight
					{
						ggml_init_params ggml_params = new ggml_init_params
						{
							mem_size = 2 * Native.ggml_tensor_overhead(),
							mem_buffer = IntPtr.Zero,
							no_alloc = true
						};
						ggml_context* ggml_context = Native.ggml_init(ggml_params);
						ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, tensor.dtype, tensor.shape.Length, tensor.shape);
						ggml_tensor->data = Marshal.AllocHGlobal(data.Length);
						Marshal.Copy(data, 0, ggml_tensor->data, data.Length);
						ggml_tensor = Native.ggml_transpose(ggml_context, ggml_tensor);
						Marshal.Copy(ggml_tensor->data, data, 0, data.Length);
						Marshal.FreeHGlobal(ggml_tensor->data);
						Native.ggml_free(ggml_context);
					}
					totalSize = totalSize + size_pad;
					if (size_pad != size)
					{
						for (long j = 0; j < size_pad - size; ++j)
						{
							data = data.Concat(new byte[] { 0 }).ToArray();
						}
					}

					using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
					{
						if (tensor.shape.Length == 1)
						{
							ushort* data_f16 = (ushort*)Marshal.AllocHGlobal(data.Length).ToPointer();
							for (int j = 0; j < data.Length; j += 2)
							{
								data_f16[j / 2] = (ushort)(data[j] | (data[j + 1] << 8));
							}
							float* data_fp32 = (float*)Marshal.AllocHGlobal(data.Length * 2).ToPointer();

							if (tensor.dtype == ggml_type.GGML_TYPE_BF16)
							{
								Native.ggml_bf16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
							}
							else if (tensor.dtype == ggml_type.GGML_TYPE_F16)
							{
								Native.ggml_fp16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
							}
							List<byte> output = new List<byte>();
							for (int j = 0; j < data.Length / 2; j++)
							{
								output.AddRange(BitConverter.GetBytes(data_fp32[j]));
							}
							Marshal.FreeHGlobal((IntPtr)data_f16);
							Marshal.FreeHGlobal((IntPtr)data_fp32);
							data = output.ToArray();
						}
						else
						{
							if (tensor.dtype == ggml_type.GGML_TYPE_BF16)
							{
								ushort* data_f16 = (ushort*)Marshal.AllocHGlobal(data.Length).ToPointer();
								for (int j = 0; j < data.Length; j += 2)
								{
									data_f16[j / 2] = (ushort)(data[j] | (data[j + 1] << 8));
								}
								float* data_fp32 = (float*)Marshal.AllocHGlobal(data.Length * 2).ToPointer();

								Native.ggml_bf16_to_fp32_row(data_f16, data_fp32, data.Length / 2);

								Native.ggml_fp32_to_fp16_row(data_fp32, data_f16, data.Length / 2);

								List<byte> output = new List<byte>();
								for (int j = 0; j < data.Length / 2; j++)
								{
									output.AddRange(BitConverter.GetBytes(data_f16[j]));
								}
								data = output.ToArray();
								Marshal.FreeHGlobal((IntPtr)data_f16);
								Marshal.FreeHGlobal((IntPtr)data_fp32);

							}
						}
						stream.Write(data, 0, data.Length);
					}
					GC.Collect();
				}
			}
			Native.gguf_free(gguf_ctx);
			Console.WriteLine("Have Done");
		}

	}
}
