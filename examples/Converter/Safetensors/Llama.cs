using Converter.Abstractions;
using GGMLSharp;
using Newtonsoft.Json.Linq;
using ProtoBuf;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using static Converter.Safetensors.SafetensorsLoader;
using static GGMLSharp.Structs;

namespace Converter.Safetensors
{
	public unsafe class Llama: ISafeTensorConverter
    {
		public void Convert(string safetensorsPath, string outputFileName, bool WriteToFileUsingStream = tru.e)
		{
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(safetensorsPath);
			Sentencepiece sentencepiece = new Sentencepiece();
			using (var fs = File.OpenRead(Path.Combine(safetensorsPath, "tokenizer.model")))
			{
				sentencepiece = Serializer.Deserialize<Sentencepiece>(fs);
			}
			gguf_context* gguf_ctx = Native.gguf_init_empty();
			string model_type = configLoader.model_type;
			Native.gguf_set_val_str(gguf_ctx, "general.architecture", model_type);
			Native.gguf_set_val_str(gguf_ctx, $"general.name", Path.GetFileName(safetensorsPath));
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.block_count", configLoader.num_hidden_layers);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.context_length", configLoader.max_position_embeddings);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.embedding_length", configLoader.hidden_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.feed_forward_length", configLoader.intermediate_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.attention.head_count", configLoader.num_attention_heads);
			Native.gguf_set_val_f32(gguf_ctx, $"{model_type}.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			Native.gguf_set_val_u32(gguf_ctx, $"general.file_type", 1);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.vocab_size", configLoader.vocab_size);
			Native.gguf_set_val_u32(gguf_ctx, $"{model_type}.rope.dimension_count", 128);
			Native.gguf_set_val_str(gguf_ctx, $"tokenizer.ggml.model", model_type);
			Native.gguf_set_val_str(gguf_ctx, $"tokenizer.ggml.pre", "default");
			List<string> tokens = new List<string>();

			sentencepiece.pieces.ForEach(piece => tokens.Add(piece.piece));

			//string k = $"tokenizer.ggml.tokens";
			//bool finded = Native.gguf_find_key(gguf_ctx, k);
			//if (!finded)
			//{
			//	int n_kv = Native.gguf_get_n_kv(gguf_ctx);

			//	gguf_ctx->kv = (gguf_kv*)Marshal.ReAllocHGlobal(new IntPtr(gguf_ctx->kv), (n_kv + 1) * sizeof(gguf_kv));
			//	gguf_ctx->kv[n_kv].key.n = (ulong)k.Length;
			//	gguf_ctx->kv[n_kv].key.data = Marshal.StringToHGlobalAnsi(k);
			//	gguf_ctx->header.n_kv++;

			//	gguf_ctx->kv[n_kv].type = gguf_type.GGUF_TYPE_ARRAY;
			//	gguf_value v = new gguf_value();
			//	v._arr.type = gguf_type.GGUF_TYPE_STRING;
			//	v._arr.n = (ulong)tokens.Count;
			//	v._arr.data = Marshal.AllocHGlobal(tokens.Count * sizeof(gguf_str));
			//	gguf_ctx->kv[n_kv].value = v;

			//	IntPtr data = Marshal.AllocHGlobal(tokens.Count * sizeof(gguf_str));
			//	for (int i = 0; i < tokens.Count; i++)
			//	{
			//		gguf_str str  = new gguf_str
			//		{
			//			n = (ulong)tokens[i].Length,
			//			data = Marshal.StringToHGlobalAnsi(tokens[i])
			//		};
			//		Marshal.StructureToPtr(str, data + i * sizeof(gguf_str), false);
			//	}
			//	gguf_ctx->kv[n_kv].value._arr.data = data;


			//}

			IntPtr[] datas = new IntPtr[tokens.Count];

			for (int i = 0; i < tokens.Count; i++)
			{
				datas[i] = Marshal.StringToCoTaskMemUTF8(tokens[i]);
			}

			Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.tokens", datas, tokens.Count);

			//Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.tokens", tokens.ToArray(), tokens.Count);

			List<float> scores = new List<float>();
			sentencepiece.pieces.ForEach(piece => scores.Add(piece.score));

			Native.gguf_set_arr_data(gguf_ctx, $"tokenizer.ggml.scores", gguf_type.GGUF_TYPE_FLOAT32, Marshal.UnsafeAddrOfPinnedArrayElement(scores.ToArray(), 0), scores.Count);
			List<int> types = new List<int>();
			sentencepiece.pieces.ForEach(piece => types.Add((int)piece.type));
			Native.gguf_set_arr_data(gguf_ctx, $"tokenizer.ggml.token_type", gguf_type.GGUF_TYPE_INT32, Marshal.UnsafeAddrOfPinnedArrayElement(types.ToArray(), 0), types.Count);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			Native.gguf_set_val_u32(gguf_ctx, $"tokenizer.ggml.padding_token_id", configLoader.pad_token_id);
			Native.gguf_set_val_bool(gguf_ctx, $"tokenizer.ggml.add_bos_token", configLoader.add_bos_token);
			Native.gguf_set_val_bool(gguf_ctx, $"tokenizer.ggml.add_eos_token", configLoader.add_eos_token);
			Native.gguf_set_val_u32(gguf_ctx, $"general.quantization_version", 2);

			//var _arr = gguf_ctx->kv[13].value._arr;
			//StringBuilder _sb = new StringBuilder();
			//for (int c = 0; c < (int)_arr.n; c++)
			//{
			//	gguf_str str = Marshal.PtrToStructure<gguf_str>(_arr.data + sizeof(gguf_str) * c);
			//	string s = Marshal.PtrToStringUTF8(str.data);
			//	_sb.AppendLine(s);
			//}
			//File.WriteAllText("token_t.txt", _sb.ToString());

			List<SafetensorsLoader.CommonTensor> tensors = new List<SafetensorsLoader.CommonTensor>();
			string[] files = Directory.GetFiles(safetensorsPath, "*.safetensors");
			foreach (string file in files)
			{
				tensors.AddRange(SafetensorsLoader.ReadTensorsInfoFromFile(file));
			}

			string header = File.ReadAllText(Path.Combine(safetensorsPath, "model.safetensors.index.json"));
			JToken token = JToken.Parse(header);
			JToken maps = token.ToObject<Dictionary<string, JToken>>().First(a => a.Key == "weight_map").Value;



			foreach ((string? key, string? value) in maps.ToObject<Dictionary<string, string>>())
			{
				Console.WriteLine($"get tensor info:{key}");
				SafetensorsLoader.CommonTensor tensor = tensors.Find(a => a.name == key);
				ggml_init_params ggml_params = new ggml_init_params
				{
					mem_size = 2 * Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};
				ggml_context* ggml_context = Native.ggml_init(ggml_params);
				long offeset = tensor.offset[0];
				int length = (int)(tensor.offset[1] - tensor.offset[0]);

				string name = SafetensorsLoader.TensorNameTrans_FromSafetensorsToGguf(tensor.name);

				if (name == tensor.name)
				{
					continue;
				}
				ggml_type type = tensor.shape.Length == 1 ? ggml_type.GGML_TYPE_F32 : ggml_type.GGML_TYPE_F16;
				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, type, tensor.shape.Length, tensor.shape);
				Native.ggml_set_name(ggml_tensor, name);
				if (!WriteToFileUsingStream)
				{
					byte[] data = ReadByteFromFile(tensor);
					if (tensor.shape.Length == 1)
					{
						ushort* data_f16 = (ushort*)Marshal.AllocHGlobal(data.Length).ToPointer();
						for (int j = 0; j < data.Length; j += 2)
						{
							data_f16[j / 2] = (ushort)(data[j] | (data[j + 1] << 8));
						}
						float* data_fp32 = (float*)Marshal.AllocHGlobal(data.Length * 2).ToPointer();

						if (tensor.dtype == ggml_type.GGML_TYPE_F16)
						{
							Native.ggml_fp16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
						}
						else if (tensor.dtype == ggml_type.GGML_TYPE_BF16)
						{
							Native.ggml_bf16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
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
							ushort* data_bf16 = (ushort*)Marshal.AllocHGlobal(data.Length).ToPointer();
							for (int j = 0; j < data.Length; j += 2)
							{
								data_bf16[j / 2] = (ushort)(data[j] | (data[j + 1] << 8));
							}
							float* data_fp32 = (float*)Marshal.AllocHGlobal(data.Length * 2).ToPointer();

							Native.ggml_bf16_to_fp32_row(data_bf16, data_fp32, data.Length / 2);

							ushort* data_fp16 = (ushort*)Marshal.AllocHGlobal(data.Length).ToPointer();
							Native.ggml_fp32_to_fp16_row(data_fp32, data_fp16, data.Length / 2);

							List<byte> output = new List<byte>();
							for (int j = 0; j < data.Length / 2; j++)
							{
								output.AddRange(BitConverter.GetBytes(data_fp16[j]));
							}
							Marshal.FreeHGlobal((IntPtr)data_bf16);
							Marshal.FreeHGlobal((IntPtr)data_fp32);
							Marshal.FreeHGlobal((IntPtr)data_fp16);
							data = output.ToArray();

						}
					}
					ggml_tensor->data = Marshal.AllocHGlobal(length);
					Marshal.Copy(data, 0, ggml_tensor->data, length);
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

			if (!WriteToFileUsingStream)
			{
				Console.WriteLine("Write to file using gguf_write_to_file function. Please wait ...");
				Native.gguf_write_to_file(gguf_ctx, "model.gguf", false);
			}
			else
			{
				Console.WriteLine("Write to file using stream.");
				string inputFileName = Path.Combine(safetensorsPath, "model.safetensors");

				Native.gguf_write_to_file(gguf_ctx, outputFileName, true);
				byte[] bytes = File.ReadAllBytes(outputFileName);
				long totalSize = bytes.Length;
				for (int i = 0; i < (int)gguf_ctx->header.n_tensors; ++i)
				{
					gguf_tensor_info* info = &gguf_ctx->infos[i];
					string name = Marshal.PtrToStringUTF8(info->name.data);
					Console.WriteLine($"{name} is doing, current total byte is {totalSize}");

					CommonTensor tensor = tensors.Find(x => SafetensorsLoader.TensorNameTrans_FromSafetensorsToGguf(x.name) == name);
					long size = Math.Max(info->size, (int)gguf_ctx->alignment);
					long _offset = tensor.offset[1] - tensor.offset[0];

					long size_pad = Native.GGML_PAD((int)size, (int)gguf_ctx->alignment);

					byte[] data = ReadByteFromFile(tensor);

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

							if (tensor.dtype == ggml_type.GGML_TYPE_F16)
							{
								Native.ggml_fp16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
							}
							else if (tensor.dtype == ggml_type.GGML_TYPE_BF16)
							{
								Native.ggml_bf16_to_fp32_row(data_f16, data_fp32, data.Length / 2);
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
								Marshal.FreeHGlobal((IntPtr)data_f16);
								Marshal.FreeHGlobal((IntPtr)data_fp32);
								data = output.ToArray();

							}
						}
						stream.Write(data, 0, data.Length);
						GC.Collect();
					}
					GC.Collect();
				}
			}
			Native.gguf_free(gguf_ctx);
			Console.WriteLine("Have Done");
		}



	}
}

