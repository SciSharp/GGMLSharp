using GGMLSharp;
using Newtonsoft.Json.Linq;
using System.Text;
using static GGMLSharp.Structs;

namespace ModelLoader
{
	public class SafetensorsLoader : IModelLoader
	{
		public List<Tensor> ReadTensorsInfoFromFile(string inputFileName)
		{
			using (FileStream stream = File.OpenRead(inputFileName))
			{
				long len = stream.Length;
				if (len < 10)
				{
					throw new ArgumentOutOfRangeException("File cannot be valid safetensors: too short");
				}

				// Safetensors file first 8 byte to int64 is the header length
				byte[] headerBlock = new byte[8];
				stream.Read(headerBlock, 0, 8);
				long headerSize = BitConverter.ToInt64(headerBlock);
				if (len < 8 + headerSize || headerSize <= 0 || headerSize > 100_000_000)
				{
					throw new ArgumentOutOfRangeException($"File cannot be valid safetensors: header len wrong, size:{headerSize}");
				}

				// Read the header, header file is a json file
				byte[] headerBytes = new byte[headerSize];
				stream.Read(headerBytes, 0, (int)headerSize);

				string header = Encoding.UTF8.GetString(headerBytes);
				long bodyPosition = stream.Position;
				JToken token = JToken.Parse(header);
				ggml_init_params @params = new ggml_init_params
				{
					mem_size = (ulong)(token.Children().Count() + 1) * Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};

				List<Tensor> tensors = new List<Tensor>();
				foreach ((string? key, JToken? subToken) in token.ToObject<Dictionary<string, JToken>>())
				{
					Dictionary<string, JToken> value = subToken.ToObject<Dictionary<string, JToken>>();
					value.TryGetValue("data_offsets", out JToken offsets);
					value.TryGetValue("dtype", out JToken dtype);
					value.TryGetValue("shape", out JToken shape);

					ulong[] offsetArray = offsets?.ToObject<ulong[]>();
					if (null == offsetArray)
					{
						continue;
					}
					long[] shapeArray = shape.ToObject<long[]>();
					if (shapeArray.Length < 1)
					{
						shapeArray = [1];
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

					Tensor tensor = new Tensor
					{
						Name = key,
						Type = ggml_type,
						Shape = shapeArray.ToList(),
						Offset = offsetArray.ToList(),
						FileName = inputFileName,
						BodyPosition = bodyPosition
					};

					tensors.Add(tensor);
				}
				return tensors;
			}
		}

		private byte[] ReadByteFromFile(string inputFileName, long bodyPosition, long offset, int size)
		{
			using (FileStream stream = File.OpenRead(inputFileName))
			{
				stream.Seek(bodyPosition + offset, SeekOrigin.Begin);
				byte[] dest = new byte[size];
				stream.Read(dest, 0, size);
				return dest;
			}
		}

		public byte[] ReadByteFromFile(Tensor tensor)
		{
			string inputFileName = tensor.FileName;
			long bodyPosition = tensor.BodyPosition;
			ulong offset = tensor.Offset[0];
			int size = (int)(tensor.Offset[1] - tensor.Offset[0]);
			return ReadByteFromFile(inputFileName, bodyPosition, (long)offset, size);
		}
	}
}
