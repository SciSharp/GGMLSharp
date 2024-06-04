using System.IO.Compression;
using static GGMLSharp.Structs;

namespace Converter
{
	internal unsafe class CommonCkptTensorReaderDemo
	{
		private class CommonTensor
		{
			public string Name { get; set; }
			public ggml_type Type { get; set; } = ggml_type.GGML_TYPE_F16;
			public long[] Shape { get; set; }
			public long[] Stride { get; set; }
			public string DataFileName { get; set; }
		}

		List<CommonTensor> tensors = new List<CommonTensor>();

		public void ReaReadTensorInfoFromFile(string fileName)
		{
			ZipArchive zip = ZipFile.OpenRead(fileName);
			ZipArchiveEntry headerEntry = zip.Entries.First(e => e.Name == "data.pkl");
			byte[] headerBytes = new byte[headerEntry.Length];
			// Header is always small enough to fit in memory, so we can read it all at once
			using (Stream stream = headerEntry.Open())
			{
				stream.Read(headerBytes, 0, headerBytes.Length);
			}

			headerBytes = headerBytes;

			if (headerBytes[0] != 0x80 || headerBytes[1] != 0x02)
			{
				throw new ArgumentException("Not a valid pickle file");
			}

			gguf_context* gguf_Context = null;

			int index = 1;
			bool finished = false;
			bool canReadShape = true;
			bool tensorsStarted = false;
			ggml_type tensorType = ggml_type.GGML_TYPE_F16;

			CommonTensor tensor = new CommonTensor();

			while (index < headerBytes.Length && !finished)
			{
				byte opcode = headerBytes[index];
				if (index == 82880)
				{
					Console.WriteLine($"opcode:{opcode}");
				}
				switch (opcode)
				{
					case (byte)'}':  // EMPTY_DICT     = b'}'   # push empty dict
						break;
					case (byte)']':  // EMPTY_LIST     = b']'   # push empty list
						break;
					// skip unused sections
					case (byte)'h':  // BINGET         = b'h'   #   "    "    "    "   "   "  ;   "    " 1-byte arg
					case (byte)'q':  // BINPUT         = b'q'   #   "     "    "   "   " ;   "    " 1-byte arg
					case (byte)'Q':  // BINPERSID      = b'Q'   #  "       "         "  ;  "  "   "     "  stack
						index++;
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
						tensorsStarted = true;
						break;
					case (byte)'K':  // BININT1        = b'K'   # push 1-byte unsigned int
						{
							int value = headerBytes[index + 1];
							index++;
							if (canReadShape)
							{
								if (tensor.Shape == null)
									tensor.Shape = new long[] { value };
								else
									tensor.Shape = tensor.Shape.Append(value).ToArray();
							}
						}
						break;
					case (byte)'M':  // BININT2        = b'M'   # push 2-byte unsigned int
						{
							int value = headerBytes[index + 2] << 8 + headerBytes[1];
							index += 2;
						}
						break;
					case (byte)'J':  // BININT         = b'J'   # push four-byte signed int
						{
							int value = headerBytes[index + 4] << 24 + headerBytes[index + 3] << 16 + headerBytes[index + 2] << 8 + headerBytes[index + 1];
							index += 4;
						}
						break;

					case (byte)'X':  // BINUNICODE     = b'X'   #   "     "       "  ; counted UTF-8 string argument
						{
							int length = headerBytes[index + 1];
							int start = index + 5;
							byte module = headerBytes[index + 1];
							string name = System.Text.Encoding.UTF8.GetString(headerBytes, start, length);
							index = index + 4 + length;

							if (tensorsStarted)
								if (string.IsNullOrEmpty(tensor.Name))
									tensor.Name = name;
							int id = zip.Entries.ToList().FindIndex(e => e.Name == name);
							if (id > -1)
							{
								tensor.DataFileName = name;
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
							if (global.Contains("FloatStorage"))
							{
								tensorType = ggml_type.GGML_TYPE_F32;
							}
							else if (global.Contains("HalfStorage"))
							{
								tensorType = ggml_type.GGML_TYPE_F16;
							}
							tensor.Type = tensorType;

							break;
						}
					case 0x86:  // TUPLE2         = b'\x86'  # build 2-tuple from two topmost stack items
					case 0x85:  // TUPLE1         = b'\x85'  # build 1-tuple from stack top
						break;
					case (byte)'t':   // TUPLE          = b't'   # build tuple from topmost stack items
									  //if (reader.phase == PickleTensorReader::READ_DIMENS)
									  //{
									  //	reader.tensor_storage.reverse_ne();
									  //	reader.tensor_storage.file_index = file_index;
									  //	// if(strcmp(prefix.c_str(), "scarlett") == 0)
									  //	// printf(" got tensor %s \n ", reader.tensor_storage.name.c_str());
									  //	reader.tensor_storage.name = prefix + reader.tensor_storage.name;
									  //	tensor_storages.push_back(reader.tensor_storage);
									  //	// LOG_DEBUG("%s", reader.tensor_storage.name.c_str());
									  //	// reset
									  //	reader = PickleTensorReader();
									  //}
						canReadShape = false;
						if (!string.IsNullOrEmpty(tensor.Name))
						{
							tensors.Add(tensor);
							tensor = new CommonTensor() { Type = tensorType };
							canReadShape = true;
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
			tensors = tensors;
		}

	}
}
