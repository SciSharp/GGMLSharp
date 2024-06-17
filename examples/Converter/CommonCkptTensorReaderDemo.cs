using GGMLSharp;
using System.IO.Compression;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace Converter
{
	internal unsafe class CommonCkptTensorReaderDemo
	{
		private class CommonTensor
		{
			public string Name { get; set; }
			public ggml_type Type { get; set; } = ggml_type.GGML_TYPE_F16;
			public List<long> Shape { get; set; } = new List<long>();
			public List<long> Stride { get; set; } = new List<long>();
			public string DataNameInZipFile { get; set; }
			public string FileName { get; set; }
		}

		private List<CommonTensor> ReadTensorInfoFromFile(string fileName)
		{
			List<CommonTensor> tensors = new List<CommonTensor>();

			ZipArchive zip = ZipFile.OpenRead(fileName);
			ZipArchiveEntry headerEntry = zip.Entries.First(e => e.Name == "data.pkl");
			byte[] headerBytes = new byte[headerEntry.Length];
			// Header is always small enough to fit in memory, so we can read it all at once
			using (Stream stream = headerEntry.Open())
			{
				stream.Read(headerBytes, 0, headerBytes.Length);
			}

			if (headerBytes[0] != 0x80 || headerBytes[1] != 0x02)
			{
				throw new ArgumentException("Not a valid pickle file");
			}

			int index = 1;
			bool finished = false;
			bool readStrides = false;
			bool binPersid = false;

			CommonTensor tensor = new CommonTensor() { FileName = fileName };

			int deepth = 0;

			Dictionary<int, string> BinPut = new Dictionary<int, string>();

			while (index < headerBytes.Length && !finished)
			{
				byte opcode = headerBytes[index];
				switch (opcode)
				{
					case (byte)'}':  // EMPTY_DICT     = b'}'   # push empty dict
						break;
					case (byte)']':  // EMPTY_LIST     = b']'   # push empty list
						break;
					// skip unused sections
					case (byte)'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
						{
							int id = headerBytes[index + 1];
							string? precision = BinPut.GetValueOrDefault(id);
							if (precision != null)
							{
								if (precision.Contains("FloatStorage"))
								{
									tensor.Type = ggml_type.GGML_TYPE_F32;
								}
								else if (precision.Contains("HalfStorage"))
								{
									tensor.Type = ggml_type.GGML_TYPE_F16;
								}
								else if (precision.Contains("BFloat16Storage"))
								{
									tensor.Type = ggml_type.GGML_TYPE_BF16;
								}
							}
							index++;
							break;
						}
					case (byte)'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
						{
							index++;
							break;
						}
					case (byte)'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
						binPersid = true;
						break;
					case (byte)'r':  // LONG_BINPUT    = b'r'   #   "     "    "   "   " ;   "    " 4-byte arg
						index += 4;
						break;
					case 0x95:  // FRAME            = b'\x95'  # indicate the beginning of a new frame
						index += 8;
						break;
					case 0x94:  // MEMOIZE          = b'\x94'  # store top of the stack in memo
						break;
					case (byte)'(':  // MARK           = b'('   # push special markobject on stack
						deepth++;
						break;
					case (byte)'K':  // BININT1        = b'K'   # push 1-byte unsigned int
						{
							int value = headerBytes[index + 1];
							index++;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									tensor.Stride.Add(value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}
						}
						break;
					case (byte)'M':  // BININT2        = b'M'   # push 2-byte unsigned int
						{
							int value = BitConverter.ToUInt16(headerBytes, index + 1);
							index += 2;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									tensor.Stride.Add(value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}
						}
						break;
					case (byte)'J':  // BININT         = b'J'   # push four-byte signed int
						{
							int value = BitConverter.ToInt32(headerBytes, index + 1);
							index += 4;

							if (deepth > 1 && value != 0 && binPersid)
							{
								if (readStrides)
								{
									tensor.Stride.Add(value);
								}
								else
								{
									tensor.Shape.Add(value);
								}
							}
						}
						break;

					case (byte)'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
						{
							int length = headerBytes[index + 1];
							int start = index + 5;
							byte module = headerBytes[index + 1];
							string name = System.Text.Encoding.UTF8.GetString(headerBytes, start, length);
							index = index + 4 + length;

							if (deepth == 1)
							{
								tensor.Name = name;
							}
							else if (deepth == 3)
							{
								if ("cpu" != name && !name.Contains("cuda"))
								{
									tensor.DataNameInZipFile = name;
								}
							}
						}
						break;
					case 0x8C:  // SHORT_BINUNICODE = b'\x8c'  # push short string; UTF-8 length < 256 bytes
						{

						}
						break;
					case (byte)'c':  // GLOBAL         = b'c'   # push self.find_class(modname, name); 2 string args
						{
							int start = index + 1;
							while (headerBytes[index + 1] != (byte)'q')
							{
								index++;
							}
							int length = index - start + 1;

							string global = System.Text.Encoding.UTF8.GetString(headerBytes, start, length);

							// precision is stored in the global variable
							// next tensor will read the precision
							// so we can set the type here

							BinPut.Add(headerBytes[index + 2], global);

							if (global.Contains("FloatStorage"))
							{
								tensor.Type = ggml_type.GGML_TYPE_F32;
							}
							else if (global.Contains("HalfStorage"))
							{
								tensor.Type = ggml_type.GGML_TYPE_F16;
							}
							else if (global.Contains("BFloat16Storage"))
							{
								tensor.Type = ggml_type.GGML_TYPE_BF16;
							}
							break;
						}
					case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
						{
							if (binPersid)
							{
								readStrides = true;
							}
							break;
						}
					case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
						if (binPersid)
						{
							readStrides = true;
						}
						break;
					case (byte)'t':   // TUPLE          = b't'   # build tuple from topmost stack items
						deepth--;
						if (binPersid)
						{
							readStrides = true;
						}
						break;
					case (byte)'R': // REDUCE         = b'R'   # apply callable to argtuple, both on stack
						if (deepth == 1)
						{
							tensors.Add(tensor);
							tensor = new CommonTensor() { FileName = fileName };
							readStrides = false;
							binPersid = false;
						}
						break;
					case (byte)'.':  // STOP           = b'.'   # every pickle ends with STOP
						finished = true;
						break;
					default:
						break;
				}
				index++;
			}
			CommonTensor? metaTensor = tensors.Find(x => x.Name.Contains("_metadata"));
			if (metaTensor != null)
			{
				tensors.Remove(metaTensor);
			}
			return tensors;
		}

		public unsafe void ConvertSafetensorsToGguf(string inputFileName, string outputFileName, bool WriteToFileUsingStream = true)
		{
			// If want to use stream to write file, set WriteToFileUsingStream to true.
			// Using gguf_write_to_file to write gguf file will read all tensors and there all data in to memory before writing file.
			// Memory usage is about 2 times of the file size. If the file is too large, it will cause out of memory.
			// Using stream to write file will avoid this problem. Memory usage is about 2 times of the largest tensor size, but not all tensors.

			List<CommonTensor> tensors = ReadTensorInfoFromFile(inputFileName);
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
				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ctx, tensors[i].Type, tensors[i].Shape.Count, tensors[i].Shape.ToArray());
				Native.ggml_set_name(ggml_tensor, tensors[i].Name);
				if (!WriteToFileUsingStream)
				{
					byte[] dest = ReadByteFromFile(tensors[i]);
					ggml_tensor->data = Marshal.AllocHGlobal(dest.Length);
					Marshal.Copy(dest, 0, ggml_tensor->data, dest.Length);
				}
				Native.gguf_add_tensor(g_ctx, ggml_tensor);
				Native.ggml_free(ctx);
				Marshal.FreeHGlobal(ggml_tensor->data);
				GC.Collect();
			}

			if (!WriteToFileUsingStream)
			{
				Native.gguf_write_to_file(g_ctx, outputFileName, false);
			}
			else
			{
				Native.gguf_write_to_file(g_ctx, outputFileName, true);

				byte[] bytes = File.ReadAllBytes(outputFileName);
				ulong totalSize = 0;
				for (int i = 0; i < (int)g_ctx->header.n_tensors; ++i)
				{
					gguf_tensor_info* info = &g_ctx->infos[i];
					string name = Marshal.PtrToStringUTF8(info->name.data);
					Console.WriteLine($"{name} is doing, current total byte is {totalSize}");

					CommonTensor tensor = tensors.Find(x => x.Name == name);
					ulong size = Math.Max(info->size, g_ctx->alignment);
					ulong size_pad = (ulong)Native.GGML_PAD((int)size, (int)g_ctx->alignment);
					byte[] data = ReadByteFromFile(tensor);
					totalSize = totalSize + size_pad;
					if (size_pad != size)
					{
						for (ulong j = 0; j < size_pad - size; ++j)
						{
							data = data.Concat(new byte[] { 0 }).ToArray();
						}
					}

					using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
					{
						stream.Write(data, 0, data.Length);
					}
					GC.Collect();
				}
			}
			Native.gguf_free(g_ctx);
			Console.WriteLine("Have Done.");

		}

		private byte[] ReadByteFromFile(CommonTensor tensor)
		{
			ZipArchive zip = ZipFile.OpenRead(tensor.FileName);
			ZipArchiveEntry dataEntry = zip.Entries.First(e => e.Name == tensor.DataNameInZipFile);
			byte[] data = new byte[dataEntry.Length];
			using (Stream stream = dataEntry.Open())
			{
				stream.Read(data, 0, data.Length);
			}
			return data;
		}

	}
}
