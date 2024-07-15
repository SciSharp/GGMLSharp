using Converter.Abstractions;
using Converter.CommonLib;
using GGMLSharp;
using ProtoBuf;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using static GGMLSharp.Structs;

namespace Converter.Safetensors
{
	internal class Llama : ISafeTensorConverter
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
			SafeGGufContext ggufContext = SafeGGufContext.Initialize();
			string modelType = configLoader.model_type;
			ggufContext.SetValueString("general.architecture", modelType);
			ggufContext.SetValueString("general.name", Path.GetFileName(folderPath));
			ggufContext.SetValueUInt32($"{modelType}.block_count", configLoader.num_hidden_layers);
			ggufContext.SetValueUInt32($"{modelType}.context_length", configLoader.max_position_embeddings);
			ggufContext.SetValueUInt32($"{modelType}.embedding_length", configLoader.hidden_size);
			ggufContext.SetValueUInt32($"{modelType}.feed_forward_length", configLoader.intermediate_size);
			ggufContext.SetValueUInt32($"{modelType}.attention.head_count", configLoader.num_attention_heads);
			ggufContext.SetValueFloat($"{modelType}.attention.layer_norm_rms_epsilon", configLoader.rms_norm_eps);
			ggufContext.SetValueUInt32($"general.file_type", 1);
			ggufContext.SetValueUInt32($"{modelType}.vocab_size", configLoader.vocab_size);
			ggufContext.SetValueUInt32($"{modelType}.rope.dimension_count", 128);
			ggufContext.SetValueString($"tokenizer.ggml.model", modelType);
			ggufContext.SetValueString($"tokenizer.ggml.pre", "default");
			List<string> tokens = new List<string>();
			sentencepiece.pieces.ForEach(piece => tokens.Add(piece.piece));
			ggufContext.SetValueArrayString($"tokenizer.ggml.tokens", tokens.ToArray());

			List<float> scores = new List<float>();
			sentencepiece.pieces.ForEach(piece => scores.Add(piece.score));

			ggufContext.SetValueArrayData($"tokenizer.ggml.scores", GGufType.GGUF_TYPE_FLOAT32, scores.ToArray());
			List<int> types = new List<int>();
			sentencepiece.pieces.ForEach(piece => types.Add((int)piece.type));
			ggufContext.SetValueArrayData($"tokenizer.ggml.token_type", GGufType.GGUF_TYPE_INT32, types.ToArray());
			ggufContext.SetValueUInt32($"tokenizer.ggml.bos_token_id", configLoader.bos_token_id);
			ggufContext.SetValueUInt32($"tokenizer.ggml.eos_token_id", configLoader.eos_token_id);
			ggufContext.SetValueUInt32($"tokenizer.ggml.padding_token_id", configLoader.pad_token_id);
			ggufContext.SetValueBool($"tokenizer.ggml.add_bos_token", configLoader.add_bos_token);
			ggufContext.SetValueBool($"tokenizer.ggml.add_eos_token", configLoader.add_eos_token);
			ggufContext.SetValueUInt32($"general.quantization_version", 2);

			List<ModelLoader.Tensor> tensors = new List<ModelLoader.Tensor>();
			ModelLoader.IModelLoader modelLoader = new ModelLoader.SafetensorsLoader();
			string[] files = Directory.GetFiles(folderPath, "*.safetensors");
			foreach (string file in files)
			{
				tensors.AddRange(modelLoader.ReadTensorsInfoFromFile(file));
			}
			Console.WriteLine("Start to load tensors from file and add to gguf_context.");
			foreach (var tensor in tensors)
			{
				Console.WriteLine($"get tensor info:{tensor.Name}");
				SafeGGmlContext ggmlContext = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true);
				long offeset = (long)tensor.Offset[0];
				int length = (int)(tensor.Offset[1] - tensor.Offset[0]);

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
					ggmlTensor.SetData(data);
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
				string inputFileName = Path.Combine(folderPath, "model.safetensors");

				ggufContext.Save(outputFileName, true);
				byte[] bytes = File.ReadAllBytes(outputFileName);
				ulong totalSize = (ulong)bytes.Length;
				for (int i = 0; i < (int)ggufContext.TensorsCount; ++i)
				{
					SafeGGufTensorInfo info = ggufContext.GGufTensorInfos[i];
					string name = info.Name;
					ModelLoader.Tensor tensor = tensors.Find(x => DataTrans.TensorNameTransToGgufName(x.Name) == name);
					ulong size = Math.Max(info.Size, ggufContext.Alignment);

					ulong size_pad = (ulong)SafeGGmlContext.GetPad((int)size, (int)ggufContext.Alignment);

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
							SafeGGmlContext ggml_context = new SafeGGmlContext(IntPtr.Zero, 2 * Common.TensorOverheadLength, true);
							SafeGGmlTensor ggml_tensor = new SafeGGmlTensor(ggml_context, tensor.Type, tensor.Shape.ToArray());
							ggml_tensor.SetData(data);
							ggml_tensor = ggml_context.Transpose(ggml_tensor);
							Marshal.Copy(ggml_tensor.Data, data, 0, data.Length);
							Marshal.FreeHGlobal(ggml_tensor.Data);
							ggml_context.Free();
						}
					}

					totalSize = totalSize + size_pad;


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
						if ((int)size_pad != data.Length)
						{
							data = data.Concat(new byte[(int)size_pad - data.Length]).ToArray();
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

