using Converter.Abstractions;
using Converter.CommonLib;
using GGMLSharp;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace Converter.Safetensors
{
	public class Qwen : ISafeTensorConverter
	{
		public void Convert(string folderPath, string outputFileName, bool WriteToFileUsingStream = true)
		{
			Console.WriteLine("Start to load configs and add to gguf_context.");
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(folderPath);
			SafeGGufContext ggufContext = SafeGGufContext.Initialize();
			ggufContext.SetValueString("general.architecture", "qwen2");
			ggufContext.SetValueString("general.name", "1_5");
			ggufContext.SetValueUInt32("qwen2.block_count", configLoader.num_hidden_layers);
			ggufContext.SetValueUInt32("qwen2.context_length", configLoader.max_position_embeddings);
			ggufContext.SetValueUInt32("qwen2.embedding_length", configLoader.hidden_size);
			ggufContext.SetValueUInt32("qwen2.feed_forward_length", configLoader.intermediate_size);
			ggufContext.SetValueUInt32("qwen2.attention.head_count", configLoader.num_attention_heads);
			ggufContext.SetValueUInt32("qwen2.attention.head_count_kv", configLoader.num_key_value_heads);
			ggufContext.SetValueFloat("qwen2.rope.freq_base", configLoader.rope_theta);
			ggufContext.SetValueFloat("qwen2.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			ggufContext.SetValueUInt32("general.file_type", 1);
			ggufContext.SetValueString("tokenizer.ggml.model", "gpt2");
			ggufContext.SetValueString("tokenizer.ggml.pre", "qwen2");
			ggufContext.SetValueArrayString("tokenizer.ggml.tokens", configLoader.tokenizer_ggml_tokens.ToArray());
			ggufContext.SetValueArrayData("tokenizer.ggml.token_type", Structs.GGufType.GGUF_TYPE_UINT32, configLoader.tokenizer_ggml_token_type.ToArray());
			ggufContext.SetValueArrayString("tokenizer.ggml.merges", configLoader.tokenizer_ggml_merges.ToArray());

			ggufContext.SetValueUInt32("tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			ggufContext.SetValueUInt32("tokenizer.ggml.padding_token_id", configLoader.pad_token_id);
			ggufContext.SetValueUInt32("tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			ggufContext.SetValueString("tokenizer.chat_template", configLoader.chat_template);

			List<ModelLoader.Tensor> tensors = new List<ModelLoader.Tensor>();
			ModelLoader.IModelLoader modelLoader = new ModelLoader.SafetensorsLoader();
			string[] files = Directory.GetFiles(folderPath, "*.safetensors");
			foreach (string file in files)
			{
				tensors.AddRange(modelLoader.ReadTensorsInfoFromFile(file));
			}

			Console.WriteLine("Start to load tensors from file and add to gguf_context.");
			foreach (ModelLoader.Tensor tensor in tensors)
			{
				Console.WriteLine($"tensor name:{tensor.Name}");
				SafeGGmlContext ggmlContext = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true);
				int length = (int)(tensor.Offset[1] - tensor.Offset[0]);

				string name = DataTrans.TensorNameTransToGgufName(tensor.Name);
				Structs.GGmlType type = tensor.Shape.Count == 1 ? Structs.GGmlType.GGML_TYPE_F32 : Structs.GGmlType.GGML_TYPE_F16;
				SafeGGmlTensor ggmlTensor = new SafeGGmlTensor(ggmlContext, type, tensor.Shape.ToArray());
				ggmlTensor.Name = name;

				if (name == tensor.Name)
				{
					continue;
				}

				if (!WriteToFileUsingStream)
				{
					byte[] tensorBytes = modelLoader.ReadByteFromFile(tensor);
					if (tensor.Shape.Count == 1)
					{
						if (tensor.Type == Structs.GGmlType.GGML_TYPE_F16)
						{
							DataConverter.Fp16ToFp32Bytes(ref tensorBytes);
						}
						else if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
						{
							DataConverter.Bf16ToFp32Bytes(ref tensorBytes);
						}
					}
					else
					{
						if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
						{
							DataConverter.Bf16ToFp16Bytes(tensorBytes);
						}
					}
					ggmlTensor.SetData(tensorBytes);
				}
				if (tensor.Shape.Count > 1)
				{
					if (name == "token_embd.Weight" || name == "output.Weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).Weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).Weight"))
					{
						ggmlTensor = ggmlContext.Transpose(ggmlTensor);
						ggmlTensor.Name = name;
					}
				}

				ggufContext.AddTensor(ggmlTensor);
				ggmlContext.Free();
				GC.Collect();
			}

			Console.WriteLine("Add to gguf_context done.");
			Console.WriteLine("Start to write gguf_context to file.");

			if (!WriteToFileUsingStream)
			{
				Console.WriteLine("Write to file using gguf_write_to_file function. Please wait ...");
				ggufContext.WriteToFile(outputFileName, false);
			}
			else
			{
				Console.WriteLine("Write to file using stream.");
				string inputFileName = Path.Combine(folderPath, "model.safetensors");
				//string outputFileName = "model.gguf";

				ggufContext.WriteToFile(outputFileName, true);

				byte[] bytes = File.ReadAllBytes(outputFileName);

				ulong totalSize = 0;
				for (int i = 0; i < (int)ggufContext.TensorsCount; ++i)
				{
					SafeGGufTensorInfo info = ggufContext.GGufTensorInfos[i];
					string name = info.Name;
					ModelLoader.Tensor tensor = tensors.Find(x => DataTrans.TensorNameTransToGgufName(x.Name) == name);
					ulong size = Math.Max(info.Size, ggufContext.Alignment);

					ulong sizePad = (ulong)SafeGGmlContext.GetPad((int)size, (int)ggufContext.Alignment);

					byte[] data = modelLoader.ReadByteFromFile(tensor);
					Console.WriteLine($"{name} is doing, bytes to read is {data.Length}, total bytes is {totalSize}");
					string transName = DataTrans.TensorNameTransToGgufName(tensor.Name);

					if (transName == tensor.Name)
					{
						continue;
					}
					if (tensor.Shape.Count > 1)
					{
						if (transName == "token_embd.Weight" || transName == "output.Weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).Weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).Weight")) //'blk.0.ffn_down.Weight
						{
							SafeGGmlContext ggml_context = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true);
							SafeGGmlTensor ggml_tensor = new SafeGGmlTensor(ggml_context, tensor.Type, tensor.Shape.ToArray());
							ggml_tensor.SetData(data);
							ggml_tensor = ggml_context.Transpose(ggml_tensor);
							Marshal.Copy(ggml_tensor.Data, data, 0, data.Length);
							Marshal.FreeHGlobal(ggml_tensor.Data);
							ggml_context.Free();
						}
					}
					totalSize = totalSize + sizePad;


					using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
					{
						if (tensor.Shape.Count == 1)
						{
							if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
							{
								DataConverter.Bf16ToFp32Bytes(ref data);
							}
							else if (tensor.Type == Structs.GGmlType.GGML_TYPE_F16)
							{
								DataConverter.Fp16ToFp32Bytes(ref data);
							}
						}
						else
						{
							if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
							{
								DataConverter.Bf16ToFp16Bytes(data);
							}
						}
						if ((int)sizePad != data.Length)
						{
							data = data.Concat(new byte[(int)sizePad - data.Length]).ToArray();
						}
						stream.Write(data, 0, data.Length);
					}
					GC.Collect();
				}
			}
			ggufContext.Free();
			Console.WriteLine("Have Done");
			Console.ReadKey();
		}

	}
}
