using Converter.Abstractions;
using Converter.CommonLib;
using GGMLSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

namespace Converter.Pickle
{
	internal class Llama3 : IPickleConverter
	{
		public void Convert(string folderPath, string outputFileName, bool WriteToFileUsingStream = true)
		{
			ConfigLoader configLoader = new ConfigLoader();
			configLoader.LoadFromFolder(folderPath);

			SafeGGufContext ggufContext = SafeGGufContext.Initialize();

			string modelType = configLoader.model_type;

			ggufContext.SetValueString("general.architecture", modelType);
			ggufContext.SetValueString("general.name", Path.GetFileName(folderPath));
			ggufContext.SetValueUInt32($"{modelType}.block_count", configLoader.num_hidden_layers);
			ggufContext.SetValueUInt32($"{modelType}.context_length", configLoader.max_position_embeddings);
			ggufContext.SetValueUInt32($"{modelType}.embedding_length", configLoader.hidden_size);
			ggufContext.SetValueUInt32($"{modelType}.feed_forward_length", configLoader.intermediate_size);
			ggufContext.SetValueUInt32($"{modelType}.attention.head_count", configLoader.num_attention_heads);
			ggufContext.SetValueUInt32($"{modelType}.attention.head_count_kv", configLoader.num_key_value_heads);
			ggufContext.SetValueFloat($"{modelType}.rope.freq_base", configLoader.rope_theta);
			ggufContext.SetValueFloat($"{modelType}.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			ggufContext.SetValueUInt32("general.file_type", 1);
			ggufContext.SetValueUInt32($"{modelType}.vocab_size", configLoader.vocab_size);
			ggufContext.SetValueUInt32($"{modelType}.rope.dimension_count", 128);
			ggufContext.SetValueString("tokenizer.ggml.model", "gpt2");
			ggufContext.SetValueString("tokenizer.ggml.pre", "smaug-bpe");
			ggufContext.SetValueArrayString("tokenizer.ggml.tokens", configLoader.tokenizer_ggml_tokens.ToArray());
			ggufContext.SetValueArrayData("tokenizer.ggml.token_type", Structs.GGufType.GGUF_TYPE_INT32, configLoader.tokenizer_ggml_token_type.ToArray());
			ggufContext.SetValueArrayString("tokenizer.ggml.merges", configLoader.tokenizer_ggml_merges.ToArray());
			ggufContext.SetValueUInt32("tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			ggufContext.SetValueUInt32("tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			ggufContext.SetValueUInt32("tokenizer.ggml.padding_token_id", configLoader.pad_token_id == 0 ? configLoader.eos_token_id : configLoader.pad_token_id);
			ggufContext.SetValueString("tokenizer.chat_template", configLoader.chat_template);
			ggufContext.SetValueUInt32("general.quantization_version", 2);

			ModelLoader.IModelLoader modelLoader = new ModelLoader.PickleLoader();
			List<ModelLoader.Tensor> tensors = new List<ModelLoader.Tensor>();
			string[] files = Directory.GetFiles(folderPath, "*.bin");
			foreach (string file in files)
			{
				tensors.AddRange(modelLoader.ReadTensorsInfoFromFile(file));
			}

			foreach (var tensor in tensors)
			{
				Console.WriteLine($"get tensor info:{tensor.Name}");

				SafeGGmlContext ggmlContext = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true);

				string name = DataTrans.TensorNameTransToGgufName(tensor.Name);

				if (name == tensor.Name)
				{
					continue;
				}
				Structs.GGmlType type = tensor.Shape.Count == 1 ? Structs.GGmlType.GGML_TYPE_F32 : Structs.GGmlType.GGML_TYPE_F16;
				SafeGGmlTensor ggmlTensor = new SafeGGmlTensor(ggmlContext, type, tensor.Shape.ToArray());
				ggmlTensor.Name = name;
				if (!WriteToFileUsingStream)
				{
					byte[] data = modelLoader.ReadByteFromFile(tensor);
					if (tensor.Shape.Count == 1)
					{
						if (tensor.Type == Structs.GGmlType.GGML_TYPE_F16)
						{
							DataConverter.Fp16ToFp32Bytes(ref data);
						}
						else if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
						{
							DataConverter.Bf16ToFp32Bytes(ref data);
						}
					}
					else
					{
						if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
						{
							DataConverter.Bf16ToFp16Bytes(data);
						}
					}
					Marshal.Copy(data, 0, ggmlTensor.Data, data.Length);
				}
				if (tensor.Shape.Count > 1)
				{
					if (name == "token_embd.weight" || name == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).weight"))
					{
						ggmlTensor = ggmlContext.Transpose(ggmlTensor);
						ggmlTensor.Name = name;
					}
				}
				ggufContext.AddTensor(ggmlTensor);
				ggmlContext.Free();
				GC.Collect();
			}

			if (!WriteToFileUsingStream)
			{
				Console.WriteLine("Write to file using gguf_write_to_file function. Please wait ...");
				ggufContext.Save(outputFileName, false);
			}
			else
			{
				Console.WriteLine("Write to file using stream.");

				ggufContext.Save(outputFileName, true);
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
						if (transName == "token_embd.weight" || transName == "output.weight" || Regex.IsMatch(name, @"blk.\d+.ffn_(gate|down|up).weight") || Regex.IsMatch(name, @"blk.\d+.attn_(v|k).weight")) //'blk.0.ffn_down.weight
						{
							SafeGGmlContext ggml_context = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true); ;
							SafeGGmlTensor ggmlTensor = new SafeGGmlTensor(ggml_context, tensor.Type, tensor.Shape.ToArray());
							ggmlTensor.SetData(data);
							ggmlTensor = ggml_context.Transpose(ggmlTensor);
							Marshal.Copy(ggmlTensor.Data, data, 0, data.Length);
							Marshal.FreeHGlobal(ggmlTensor.Data);
							ggml_context.Free();
						}
					}

					totalSize = totalSize + sizePad;

					using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
					{
						if (tensor.Shape.Count == 1)
						{
							if (tensor.Type == Structs.GGmlType.GGML_TYPE_F16)
							{
								DataConverter.Fp16ToFp32Bytes(ref data);
							}
							else if (tensor.Type == Structs.GGmlType.GGML_TYPE_BF16)
							{
								DataConverter.Bf16ToFp32Bytes(ref data);
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
			Console.Read();


		}
	}
}
