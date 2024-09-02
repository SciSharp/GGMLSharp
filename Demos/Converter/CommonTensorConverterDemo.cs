using GGMLSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Converter
{
	internal class CommonTensorConverterDemo
	{
		public enum ModelType
		{
			Safetensors,
			Pickle,
		}

		public void ConvertToGguf(string inputFileName, string outputFileName, ModelType modelType, bool writeToFileUsingStream)
		{
			// If want to use stream to write file, set writeToFileUsingStream to true.
			// Using gguf_write_to_file to write gguf file will read all tensors and there all data in to memory before writing file.
			// Memory usage is about 2 times of the file size. If the file is too large, it will cause out of memory.
			// Using stream to write file will avoid this problem. Memory usage is about 2 times of the largest tensor size, but not all tensors.

			ModelLoader.IModelLoader modelLoader;
			switch (modelType)
			{
				case ModelType.Safetensors:
					modelLoader = new ModelLoader.SafetensorsLoader();
					break;
				case ModelType.Pickle:
					modelLoader = new ModelLoader.PickleLoader();
					break;
				default:
					throw new ArgumentException("Model Type Not Support");
			}
			List<ModelLoader.Tensor> tensors = modelLoader.ReadTensorsInfoFromFile(inputFileName);
			SafeGGufContext ggufContext = SafeGGufContext.Initialize();

			for (int i = 0; i < tensors.Count; i++)
			{
				SafeGGmlContext ggmlContext = new SafeGGmlContext(IntPtr.Zero, Common.TensorOverheadLength, true);

				SafeGGmlTensor ggmlTensor = new SafeGGmlTensor(ggmlContext, tensors[i].Type, tensors[i].Shape.ToArray());
				ggmlTensor.Name = tensors[i].Name;

				// If want to use stream to write file, we can read each tensors data while writing file, and free them after writing immediately.
				if (!writeToFileUsingStream)
				{
					byte[] dest = modelLoader.ReadByteFromFile(tensors[i]);
					ggmlTensor.SetData(dest);
				}
				ggufContext.AddTensor(ggmlTensor);
				ggmlContext.Free();
				GC.Collect();
			}

			if (!writeToFileUsingStream)
			{
				ggufContext.WriteToFile(outputFileName, false);
			}
			else
			{
				ggufContext.WriteToFile(outputFileName, true);

				byte[] bytes = File.ReadAllBytes(outputFileName);
				ulong totalSize = 0;
				for (int i = 0; i < (int)ggufContext.TensorsCount; ++i)
				{
					SafeGGufTensorInfo info = ggufContext.GGufTensorInfos[i];
					string name = info.Name;
					Console.WriteLine($"{name} is doing, current total byte is {totalSize}");

					ModelLoader.Tensor tensor = tensors.Find(x => x.Name == name);
					ulong size = Math.Max(info.Size, ggufContext.Alignment);

					ulong size_pad = (ulong)SafeGGmlContext.GetPad((int)size, (int)ggufContext.Alignment);

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

			ggufContext.Free();
			Console.WriteLine("Have Done.");
			Console.ReadKey();

		}


	}
}
