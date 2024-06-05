using GGMLSharp;
using Newtonsoft.Json.Linq;
using System.Text;
using System.Text.RegularExpressions;
using static GGMLSharp.Structs;

namespace Converter.Safetensors
{
	internal class SafetensorsLoader
	{
		public class CommonTensor
		{
			public string name { get; set; }
			public ggml_type dtype { get; set; }
			public long[] shape { get; set; }
			public long[] offset { get; set; }
			public string file_name { get; set; }
			public long body_position { get; set; }
		}


		public static List<CommonTensor> ReadTensorsInfoFromFile(string inputFileName)
		{
			return ReadTensorsInfoFromFile(inputFileName, out _);
		}

		private static List<CommonTensor> ReadTensorsInfoFromFile(string inputFileName, out long bodyPosition)
		{
			using (FileStream stream = File.OpenRead(inputFileName))
			{
				long len = stream.Length;
				if (len < 10)
				{
					throw new ArgumentOutOfRangeException("File cannot be valid safetensors: too short");
				}
				byte[] headerBlock = new byte[8];
				stream.Read(headerBlock, 0, 8);
				long headerSize = BitConverter.ToInt64(headerBlock);
				if (len < 8 + headerSize || headerSize <= 0 || headerSize > 100_000_000)
				{
					throw new ArgumentOutOfRangeException($"File cannot be valid safetensors: header len wrong, size:{headerSize}");
				}

				byte[] headerBytes = new byte[headerSize];
				stream.Read(headerBytes, 0, (int)headerSize);

				string header = Encoding.UTF8.GetString(headerBytes);
				bodyPosition = stream.Position;
				JToken token = JToken.Parse(header);
				ggml_init_params @params = new ggml_init_params
				{
					mem_size = (token.Children().Count() + 1) * Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};

				List<CommonTensor> tensors = new List<CommonTensor>();
				foreach ((string? key, JToken? subToken) in token.ToObject<Dictionary<string, JToken>>())
				{
					Dictionary<string, JToken> value = subToken.ToObject<Dictionary<string, JToken>>();
					value.TryGetValue("data_offsets", out JToken offsets);
					value.TryGetValue("dtype", out JToken dtype);
					value.TryGetValue("shape", out JToken shape);

					long[] offsetArray = offsets?.ToObject<long[]>();
					if (null == offsetArray)
					{
						continue;
					}
					long[] shapeArray = shape.ToObject<long[]>();
					if (shapeArray.Length < 1)
					{
						shapeArray = new long[] { 1 };
					}
					ggml_type ggml_type = ggml_type.GGML_TYPE_F32;
					switch (dtype.ToString())
					{
						case "I8": ggml_type = ggml_type.GGML_TYPE_I8; break;
						case "I16": ggml_type = ggml_type.GGML_TYPE_I16; break;
						case "I32": ggml_type = ggml_type.GGML_TYPE_I32; break;
						case "I64": ggml_type = ggml_type.GGML_TYPE_I64; break;
						case "BF16": ggml_type = ggml_type.GGML_TYPE_BF16; break;
						case "F16": ggml_type = ggml_type.GGML_TYPE_F16; break;
						case "F32": ggml_type = ggml_type.GGML_TYPE_F32; break;
						case "F64": ggml_type = ggml_type.GGML_TYPE_F64; break;
						case "U8":
						case "U16":
						case "U32":
						case "U64":
						case "BOOL":
						case "F8_E4M3":
						case "F8_E5M2": break;
					}

					CommonTensor tensor = new CommonTensor
					{
						name = key,
						dtype = ggml_type,
						shape = shapeArray,
						offset = offsetArray,
						file_name = inputFileName,
						body_position = bodyPosition
					};

					tensors.Add(tensor);
				}
				return tensors;
			}
		}

		private static byte[] ReadByteFromFile(string inputFileName, long bodyPosition, long offset, int size)
		{
			using (FileStream stream = File.OpenRead(inputFileName))
			{
				stream.Seek(bodyPosition + offset, SeekOrigin.Begin);
				byte[] dest = new byte[size];
				stream.Read(dest, 0, size);
				return dest;
			}
		}

		public static byte[] ReadByteFromFile(CommonTensor tensor)
		{
			string inputFileName = tensor.file_name;
			long bodyPosition = tensor.body_position;
			long offset = tensor.offset[0];
			int size = (int)(tensor.offset[1] - tensor.offset[0]);
			return ReadByteFromFile(inputFileName, bodyPosition, offset, size);
		}

		public static string TensorNameTrans_FromSafetensorsToGguf(string inputTensorName)
		{
			if (inputTensorName == "lm_head.weight")
			{
				return "output.weight";
			}
			else if (inputTensorName == "model.embed_tokens.weight")
			{
				return "token_embd.weight";
			}
			else if (inputTensorName == "model.norm.weight")
			{
				return "output_norm.weight";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).input_layernorm.weight"))
			{
				string num = new Regex(@"model.layers.(\d+).input_layernorm.weight").Match(inputTensorName).Groups[1].Value;
				return $"blk.{num}.attn_norm.weight";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).mlp.(\w+)_proj.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).mlp.(\w+)_proj.(\w+)").Match(inputTensorName);
				return $"blk.{match.Groups[1]}.ffn_{match.Groups[2]}.{match.Groups[3]}";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).post_attention_layernorm.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).post_attention_layernorm.(\w+)").Match(inputTensorName);
				return $"blk.{match.Groups[1]}.ffn_norm.{match.Groups[2]}";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).self_attn.([kqvo])_proj.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).self_attn.([kqvo])_proj.(\w+)").Match(inputTensorName);
				if (match.Groups[2].Value == "o")
				{
					return $"blk.{match.Groups[1]}.attn_output.{match.Groups[3]}";
				}
				return $"blk.{match.Groups[1]}.attn_{match.Groups[2]}.{match.Groups[3]}";
			}

			else
			{
				return inputTensorName;
			}
		}
	}
}
