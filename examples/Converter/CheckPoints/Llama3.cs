using Converter.CommonLib;
using GGMLSharp;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using static GGMLSharp.Structs;
using Converter.Abstractions;

namespace Converter.CheckPoints
{
	internal class Llama3 : IPickleConverter
	{
		public unsafe void Convert(string folderPath, string outputFileName, bool WriteToFileUsingStream = true)
		{
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(folderPath);

			gguf_context* gguf_ctx = Native.gguf_init_empty();

			string model_type = configLoader.model_type;

			Native.gguf_set_val_str(gguf_ctx, "general.architecture", model_type);
			Native.gguf_set_val_str(gguf_ctx, $"general.name", Path.GetFileName(folderPath));
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.block_count", configLoader.num_hidden_layers);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.context_length", configLoader.max_position_embeddings);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.embedding_length", configLoader.hidden_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.feed_forward_length", configLoader.intermediate_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.attention.head_count", configLoader.num_attention_heads);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.attention.head_count_kv", configLoader.num_key_value_heads);
			Native.gguf_set_val_f32(gguf_ctx, $"{model_type}.rope.freq_base", configLoader.rope_theta);
			Native.gguf_set_val_f32(gguf_ctx, $"{model_type}.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			Native.gguf_set_val_u32(gguf_ctx, $"general.file_type", 1);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.vocab_size", configLoader.vocab_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.rope.dimension_count", 128);
			Native.gguf_set_val_str(gguf_ctx, $"tokenizer.ggml.model", "gpt2");
			Native.gguf_set_val_str(gguf_ctx, $"tokenizer.ggml.pre", "smaug-bpe");
			IntPtr[] datas = new IntPtr[configLoader.tokenizer_ggml_tokens.Count];

			for (int i = 0; i < configLoader.tokenizer_ggml_tokens.Count; i++)
			{
				datas[i] = Marshal.StringToCoTaskMemUTF8(configLoader.tokenizer_ggml_tokens[i]);
			}

			Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.tokens", datas, configLoader.tokenizer_ggml_tokens.Count);
			Native.gguf_set_arr_data(gguf_ctx, $"tokenizer.ggml.token_type", gguf_type.GGUF_TYPE_INT32, Marshal.UnsafeAddrOfPinnedArrayElement(configLoader.tokenizer_ggml_token_type.ToArray(), 0), configLoader.tokenizer_ggml_token_type.Count);
			IntPtr[] mergesData = new IntPtr[configLoader.tokenizer_ggml_merges.Count];
			for (int i = 0; i < configLoader.tokenizer_ggml_merges.Count; i++)
			{
				mergesData[i] = Marshal.StringToCoTaskMemUTF8(configLoader.tokenizer_ggml_merges[i]);
			}
			Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.merges", mergesData, configLoader.tokenizer_ggml_merges.Count);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.padding_token_id", configLoader.pad_token_id == 0 ? configLoader.eos_token_id : configLoader.pad_token_id);
			Native.gguf_set_val_str(gguf_ctx, $"tokenizer.chat_template", configLoader.chat_template);
			Native.gguf_set_val_u32(gguf_ctx, $"general.quantization_version", 2);

			List<PickleLoader.CommonTensor> tensors = new List<PickleLoader.CommonTensor>();
			string[] files = Directory.GetFiles(folderPath, "*.bin");
			foreach (string file in files)
			{
				tensors.AddRange(PickleLoader.ReadTensorInfoFromFile(file));
			}

			foreach (var tensor in tensors)
			{
				Console.WriteLine($"get tensor info:{tensor.Name}");

				ggml_init_params ggml_params = new ggml_init_params
				{
					mem_size = 2 * Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};
				ggml_context* ggml_context = Native.ggml_init(ggml_params);

				string name = CommonLib.DataTrans.TensorNameTransToGgufName(tensor.Name);

				if (name == tensor.Name)
				{
					continue;
				}
				ggml_type type = tensor.Shape.Count == 1 ? ggml_type.GGML_TYPE_F32 : ggml_type.GGML_TYPE_F16;
				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, type, tensor.Shape.Count, tensor.Shape.ToArray());
				Native.ggml_set_name(ggml_tensor, name);
				if (!WriteToFileUsingStream)
				{
					byte[] data = PickleLoader.ReadByteFromFile(tensor);
					if (tensor.Shape.Count == 1)
					{
						if (tensor.Type == ggml_type.GGML_TYPE_F16)
						{
							data = CommonLib.DataTrans.Fp16ToF32Bytes(data);
						}
						else if (tensor.Type == ggml_type.GGML_TYPE_BF16)
						{
							data = CommonLib.DataTrans.Bf16ToF32Bytes(data);
						}
					}
					else
					{
						if (tensor.Type == ggml_type.GGML_TYPE_BF16)
						{
							data = CommonLib.DataTrans.Bf16ToFp16Bytes(data);
						}
					}
					ggml_tensor->data = Marshal.AllocHGlobal(data.Length);
					Marshal.Copy(data, 0, ggml_tensor->data, data.Length);
				}
				if (tensor.Shape.Count > 1)
				{
					if (name == "token_embd.weight" || name == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).weight"))
					{
						ggml_tensor = Native.ggml_transpose(ggml_context, ggml_tensor);
						Native.ggml_set_name(ggml_tensor, name);
					}
				}
				Native.gguf_add_tensor(gguf_ctx, ggml_tensor);
				Native.ggml_free(ggml_context);
				GC.Collect();
			}

			if (!WriteToFileUsingStream)
			{
				Console.WriteLine("Write to file using gguf_write_to_file function. Please wait ...");
				Native.gguf_write_to_file(gguf_ctx, "model.gguf", false);
			}
			else
			{
				Console.WriteLine("Write to file using stream.");

				Native.gguf_write_to_file(gguf_ctx, outputFileName, true);
				byte[] bytes = File.ReadAllBytes(outputFileName);
				long totalSize = 0;
				for (int i = 0; i < (int)gguf_ctx->header.n_tensors; ++i)
				{
					gguf_tensor_info* info = &gguf_ctx->infos[i];
					string name = Marshal.PtrToStringUTF8(info->name.data);
					PickleLoader.CommonTensor tensor = tensors.Find(x => CommonLib.DataTrans.TensorNameTransToGgufName(x.Name) == name);
					long size = Math.Max(info->size, (int)gguf_ctx->alignment);

					long size_pad = Native.GGML_PAD((int)size, (int)gguf_ctx->alignment);

					byte[] data = PickleLoader.ReadByteFromFile(tensor);
					Console.WriteLine($"{name} is doing, bytes to read is {data.Length}, total bytes is {totalSize}");

					string transName = CommonLib.DataTrans.TensorNameTransToGgufName(tensor.Name);

					if (transName == tensor.Name)
					{
						continue;
					}
					if (tensor.Shape.Count > 1)
					{
						if (transName == "token_embd.weight" || transName == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).weight")) //'blk.0.ffn_down.weight
						{
							ggml_init_params ggml_params = new ggml_init_params
							{
								mem_size = 2 * Native.ggml_tensor_overhead(),
								mem_buffer = IntPtr.Zero,
								no_alloc = true
							};
							ggml_context* ggml_context = Native.ggml_init(ggml_params);
							ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, tensor.Type, tensor.Shape.Count, tensor.Shape.ToArray());
							ggml_tensor->data = Marshal.AllocHGlobal(data.Length);
							Marshal.Copy(data, 0, ggml_tensor->data, data.Length);
							ggml_tensor = Native.ggml_transpose(ggml_context, ggml_tensor);
							Marshal.Copy(ggml_tensor->data, data, 0, data.Length);
							Marshal.FreeHGlobal(ggml_tensor->data);
							Native.ggml_free(ggml_context);
						}
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
						if (tensor.Shape.Count == 1)
						{
							if (tensor.Type == ggml_type.GGML_TYPE_F16)
							{
								data = CommonLib.DataTrans.Fp16ToF32Bytes(data);
							}
							else if (tensor.Type == ggml_type.GGML_TYPE_BF16)
							{
								data = CommonLib.DataTrans.Bf16ToF32Bytes(data);
							}
						}
						else
						{
							if (tensor.Type == ggml_type.GGML_TYPE_BF16)
							{
								data = CommonLib.DataTrans.Bf16ToFp16Bytes(data);
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
