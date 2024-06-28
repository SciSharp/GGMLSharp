using GGMLSharp;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace Converter
{
	internal class CommonSafeTensorReaderDemo
	{
		public unsafe void ConvertSafetensorsToGguf(string inputFileName, string outputFileName)
		{
			// If want to use stream to write file, set WriteToFileUsingStream to true.
			// Using gguf_write_to_file to write gguf file will read all tensors and there all data in to memory before writing file.
			// Memory usage is about 2 times of the file size. If the file is too large, it will cause out of memory.
			// Using stream to write file will avoid this problem. Memory usage is about 2 times of the largest tensor size, but not all tensors.
			bool WriteToFileUsingStream = true;

			ModelLoader.IModelLoader modelLoader = new ModelLoader.SafetensorsLoader();
			List<ModelLoader.Tensor> tensors = modelLoader.ReadTensorsInfoFromFile(inputFileName);
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
				int length = (int)(tensors[i].Offset[1] - tensors[i].Offset[0]);

				ggml_tensor* ggml_tensor = Native.ggml_new_tensor(ctx, tensors[i].Type, tensors[i].Shape.Count, tensors[i].Shape.ToArray());
				Native.ggml_set_name(ggml_tensor, tensors[i].Name);

				// If want to use stream to write file, we can read each tensors data while writing file, and free them after writing immediately.
				if (!WriteToFileUsingStream)
				{
					byte[] dest = modelLoader.ReadByteFromFile(tensors[i]);
					ggml_tensor->data = Marshal.AllocHGlobal(length);
					Marshal.Copy(dest, 0, ggml_tensor->data, length);
				}
				Native.gguf_add_tensor(g_ctx, ggml_tensor);
				Native.ggml_free(ctx);
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
					string name = Marshal.PtrToStringAnsi(info->name.data);
					Console.WriteLine($"{name} is doing, current total byte is {totalSize}");

					ModelLoader.Tensor tensor = tensors.Find(x => x.Name == name);
					ulong size = Math.Max(info->size, g_ctx->alignment);

					ulong size_pad = (ulong)Native.GGML_PAD((int)size, (int)g_ctx->alignment);

					byte[] data = modelLoader.ReadByteFromFile(tensor);

					totalSize = totalSize + size_pad;

					using (FileStream stream = new FileStream(outputFileName, FileMode.Append, FileAccess.Write))
					{
						if ((int)size_pad != data.Length)
						{
							data = data.Concat(new byte[(int)size_pad - data.Length]).ToArray();
						}
						stream.Write(data, 0, data.Length);
					}
					GC.Collect();
				}
			}

			Native.gguf_free(g_ctx);

			Console.WriteLine("Have Done.");

		}


	}
}
