using Converter.Abstractions;
using Converter.CommonLib;
using GGMLSharp;
using ProtoBuf;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using static Converter.Safetensors.SafetensorsLoader;
using static GGMLSharp.Structs;

namespace Converter.Safetensors
{
	internal unsafe class Llama : ISafeTensorConverter
	{
		public void Convert(string folderPath, string outputFileName, bool WriteToFileUsingStream = true)
		{
			Console.WriteLine("Start to load configs and add to gguf_context.");
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(folderPath);
			Sentencepiece sentencepiece = new Sentencepiece();
			using (var fs = File.OpenRead(Path.Combine(folderPath, "tokenizer.model")))
			{
				sentencepiece = Serializer.Deserialize<Sentencepiece>(fs);
			}
			gguf_context* gguf_ctx = Native.gguf_init_empty();
			string model_type = configLoader.model_type;
			Native.gguf_set_val_str(gguf_ctx, "general.architecture", model_type);
			Native.gguf_set_val_str(gguf_ctx, $"general.name", Path.GetFileName(folderPath));
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

			IntPtr[] datas = new IntPtr[tokens.Count];

			for (int i = 0; i < tokens.Count; i++)
			{
				datas[i] = Marshal.StringToCoTaskMemUTF8(tokens[i]);
			}

			Native.gguf_set_arr_str(gguf_ctx, $"tokenizer.ggml.tokens", datas, tokens.Count);

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

			List<SafetensorsLoader.CommonTensor> tensors = new List<SafetensorsLoader.CommonTensor>();
			string[] files = Directory.GetFiles(folderPath, "*.safetensors");
			foreach (string file in files)
			{
				tensors.AddRange(SafetensorsLoader.ReadTensorsInfoFromFile(file));
			}
			Console.WriteLine("Start to load tensors from file and add to gguf_context.");
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
				long offeset = tensor.Offset[0];
				int length = (int)(tensor.Offset[1] - tensor.Offset[0]);

				string name = CommonLib.DataTrans.TensorNameTransToGgufName(tensor.Name);

				if (name == tensor.Name)
				{
					continue;
				}
				ggml_type type = tensor.Shape.Length == 1 ? ggml_type.GGML_TYPE_F32 : ggml_type.GGML_TYPE_F16;
				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, type, tensor.Shape.Length, tensor.Shape);
				Native.ggml_set_name(ggml_tensor, name);
				if (!WriteToFileUsingStream)
				{
					byte[] data = ReadByteFromFile(tensor);
					if (tensor.Shape.Length == 1)
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
					ggml_tensor->data = Marshal.AllocHGlobal(length);
					Marshal.Copy(data, 0, ggml_tensor->data, length);
				}
				if (tensor.Shape.Length > 1)
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
				string inputFileName = Path.Combine(folderPath, "model.safetensors");

				Native.gguf_write_to_file(gguf_ctx, outputFileName, true);
				byte[] bytes = File.ReadAllBytes(outputFileName);
				long totalSize = bytes.Length;
				for (int i = 0; i < (int)gguf_ctx->header.n_tensors; ++i)
				{
					gguf_tensor_info* info = &gguf_ctx->infos[i];
					string name = Marshal.PtrToStringUTF8(info->name.data);
					CommonTensor tensor = tensors.Find(x => CommonLib.DataTrans.TensorNameTransToGgufName(x.Name) == name);
					long size = Math.Max(info->size, (int)gguf_ctx->alignment);

					long size_pad = Native.GGML_PAD((int)size, (int)gguf_ctx->alignment);

					byte[] data = ReadByteFromFile(tensor);
					Console.WriteLine($"{name} is doing, bytes to read is {data.Length}, total bytes is {totalSize}");

					string transName = CommonLib.DataTrans.TensorNameTransToGgufName(tensor.Name);

					if (transName == tensor.Name)
					{
						continue;
					}
					if (tensor.Shape.Length > 1)
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
							ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ggml_context, tensor.Type, tensor.Shape.Length, tensor.Shape);
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
						if (tensor.Shape.Length == 1)
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

