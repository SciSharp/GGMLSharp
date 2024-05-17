using GGMLSharp;
using Newtonsoft.Json.Linq;
using System.Runtime.InteropServices;
using System.Text;
using static GGMLSharp.Structs;

namespace Converter
{
	internal class SafeTensorReader
	{
		class CommonTensor
		{
			public string name { get; set; }
			public ggml_type dtype { get; set; }
			public long[] shape { get; set; }
			public long[] offset { get; set; }
		}

		private static List<CommonTensor> ReadTensorInfoFromFile(string inputFileName, out long bodyPosition)
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
					throw new ArgumentOutOfRangeException($"File cannot be valid safetensors: header len wrong {headerSize}");
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
					};

					tensors.Add(tensor);
				}
				return tensors;
			}
		}

		private static byte[] ReadByteFromFile(string inputFileName, long bodyPosition, int offset, int size)
		{
			using (FileStream stream = File.OpenRead(inputFileName))
			{
				stream.Seek(bodyPosition + offset, SeekOrigin.Begin);
				byte[] dest = new byte[size];
				stream.Read(dest, 0, size);
				return dest;
			}
		}


		public unsafe void ConvertSafetensorsToGguf(string inputFileName, string outputFileName)
		{
			List<CommonTensor> tensors = ReadTensorInfoFromFile(inputFileName, out long bPosition);
			gguf_context* g_ctx = Native.gguf_init_empty();
			for (int i = 0; i < tensors.Count; i++)
			{
				ggml_init_params @params = new ggml_init_params
				{
					mem_size = Native.ggml_tensor_overhead(),
					mem_buffer = IntPtr.Zero,
					no_alloc = true
				};
				ggml_context* ctx = Native.ggml_init(@params);
				byte[] dest = ReadByteFromFile(inputFileName, (int)bPosition, (int)tensors[i].offset[0], (int)tensors[i].offset[1] - (int)tensors[i].offset[0]);
				fixed (long* ne = tensors[i].shape)
				{
					ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ctx, tensors[i].dtype, tensors[i].shape.Length, ne);
					Native.ggml_set_name(ggml_tensor, tensors[i].name);
					//ggml_tensor->data = Marshal.UnsafeAddrOfPinnedArrayElement(dest, 0);
					Native.gguf_add_tensor(g_ctx, ggml_tensor);
				}

				Native.ggml_free(ctx);
			}

			Native.gguf_write_to_file(g_ctx, outputFileName, true);

			byte[] bytes = File.ReadAllBytes(outputFileName);
			int totalSize = bytes.Length;
			for (int i = 0; i < (int)g_ctx->header.n_tensors; ++i)
			{
				gguf_tensor_info* info = &g_ctx->infos[i];
				string name = Marshal.PtrToStringUTF8(info->name.data);
				Console.WriteLine($"{name} is doing, current total byte is {totalSize}");

				CommonTensor tensor = tensors.Find(x => x.name == name);
				long size = Math.Max(info->size, (int)g_ctx->alignment);
				long _offset = tensor.offset[1] - tensor.offset[0];

				long size_pad = Native.GGML_PAD((int)size, (int)g_ctx->alignment);

				byte[] data = ReadByteFromFile(inputFileName, (int)bPosition, (int)tensor.offset[0], (int)size);
				totalSize = totalSize + (int)size_pad;
				if (size_pad != size)
				{
					for (long j = 0; j < size_pad - size; ++j)
					{
						data = data.Concat(new byte[] { 0 }).ToArray();
					}
				}
				//byte[] data = new byte[size];
				//Marshal.Copy(info->data, data, 0, (int)size);

				using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
				{
					stream.Write(data, 0, data.Length);
				}
				GC.Collect();
			}

			Native.gguf_free(g_ctx);

			Console.WriteLine("Have Done.");

		}

		private byte[] gguf_bwrite_el(byte[] buf, IntPtr data, int size)
		{
			byte[] bytes = new byte[size];
			Marshal.Copy(data, bytes, 0, size);
			return buf.Concat(bytes).ToArray();
		}

	}
}
