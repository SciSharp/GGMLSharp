using GGMLSharp;

namespace Magika
{
	internal class Program
	{

		static string[] Labels ={
				"ai",                 "apk",                "appleplist",         "asm",                "asp",
				"batch",              "bmp",                "bzip",               "c",                  "cab",
				"cat",                "chm",                "coff",               "crx",                "cs",
				"css",                "csv",                "deb",                "dex",                "dmg",
				"doc",                "docx",               "elf",                "emf",                "eml",
				"epub",               "flac",               "gif",                "go",                 "gzip",
				"hlp",                "html",               "ico",                "ini",                "internetshortcut",
				"iso",                "jar",                "java",               "javabytecode",       "javascript",
				"jpeg",               "json",               "latex",              "lisp",               "lnk",
				"m3u",                "macho",              "makefile",           "markdown",           "mht",
				"mp3",                "mp4",                "mscompress",         "msi",                "mum",
				"odex",               "odp",                "ods",                "odt",                "ogg",
				"outlook",            "pcap",               "pdf",                "pebin",              "pem",
				"perl",               "php",                "png",                "postscript",         "powershell",
				"ppt",                "pptx",               "python",             "pythonbytecode",     "rar",
				"rdf",                "rpm",                "rst",                "rtf",                "ruby",
				"rust",               "scala",              "sevenzip",           "shell",              "smali",
				"sql",                "squashfs",           "svg",                "swf",                "symlinktext",
				"tar",                "tga",                "tiff",               "torrent",            "ttf",
				"txt",                "unknown",            "vba",                "wav",                "webm",
				"webp",               "winregistry",        "wmf",                "xar",                "xls",
				"xlsb",               "xlsx",               "xml",                "xpi",                "xz",
				"yaml",               "zip",                "zlibstream"
			};

		private class Hparams
		{
			public int blockSize = 4096;
			public int begSize = 512;
			public int midSize = 512;
			public int endSize = 512;
			public int minFileSizeForDl = 16;
			public int labelCount = 113;
			public float normEps = 0.001f;
			public int paddingToken = 256;
		};

		private class MagikaModel
		{
			~MagikaModel()
			{
				backendBuffer.Free();
				backend.Free();
				context.Free();
			}

			public Hparams hparams = new Hparams();

			public SafeGGmlTensor denseWeight;
			public SafeGGmlTensor denseBias;

			public SafeGGmlTensor layerNormGamma;
			public SafeGGmlTensor layerNormBeta;

			public SafeGGmlTensor dense1Weight;
			public SafeGGmlTensor dense1Bias;

			public SafeGGmlTensor dense2Weight;
			public SafeGGmlTensor dense2Bias;

			public SafeGGmlTensor layerNorm1Gamma;
			public SafeGGmlTensor layerNorm1Beta;

			public SafeGGmlTensor targetLabelWeight;
			public SafeGGmlTensor targetLabelBias;

			public SafeGGmlBackend backend = SafeGGmlBackend.CpuInit();
			public SafeGGmlBackendBuffer backendBuffer;

			public SafeGGmlContext context = new SafeGGmlContext(IntPtr.Zero);
		};

		private static SafeGGmlTensor CheckedGetTensor(SafeGGmlContext ctx, string name)
		{
			SafeGGmlTensor tensor = ctx.GetTensor(name);
			if (tensor.IsInvalid)
			{
				throw new ArgumentNullException($"tensor {name} not found");
			}
			return tensor;
		}

		private static MagikaModel LoadModel(string fname)
		{
			MagikaModel model = new MagikaModel();
			SafeGGufContext ggufContext = SafeGGufContext.InitFromFile("./Assets/magika.gguf", model.context, true);

			model.backend = SafeGGmlBackend.CpuInit(); // init device 0

			if (!ggufContext.IsHeaderMagicMatch)
			{
				throw new FileLoadException("gguf_init_from_file failed");
			}
			model.backendBuffer = model.context.BackendAllocContextTensors(model.backend);
			if (model.backendBuffer.IsInvalid)
			{
				ggufContext.Free();
				throw new Exception("ggml_backend_alloc_ctx_tensors failed");
			}

			try
			{
				model.denseWeight = CheckedGetTensor(model.context, "dense/kernel:0");
				model.denseBias = CheckedGetTensor(model.context, "dense/bias:0");

				model.layerNormGamma = CheckedGetTensor(model.context, "layer_normalization/gamma:0");
				model.layerNormBeta = CheckedGetTensor(model.context, "layer_normalization/beta:0");

				model.dense1Weight = CheckedGetTensor(model.context, "dense_1/kernel:0");
				model.dense1Bias = CheckedGetTensor(model.context, "dense_1/bias:0");

				model.dense2Weight = CheckedGetTensor(model.context, "dense_2/kernel:0");
				model.dense2Bias = CheckedGetTensor(model.context, "dense_2/bias:0");

				model.layerNorm1Gamma = CheckedGetTensor(model.context, "layer_normalization_1/gamma:0");
				model.layerNorm1Beta = CheckedGetTensor(model.context, "layer_normalization_1/beta:0");

				model.targetLabelWeight = CheckedGetTensor(model.context, "target_label/kernel:0");
				model.targetLabelBias = CheckedGetTensor(model.context, "target_label/bias:0");
			}
			catch (Exception ex)
			{
				Console.WriteLine(ex.Message);
				ggufContext.Free();
				return null;
			}

			using (FileStream fs = new FileStream(fname, FileMode.Open, FileAccess.Read))
			{
				for (ulong i = 0; i < ggufContext.TensorsCount; i++)
				{
					string name = ggufContext.GetTensorName((int)i);

					SafeGGmlTensor tensor = model.context.GetTensor(name);
					ulong offs = ggufContext.GetDataOffset() + ggufContext.GetTensorOffset((int)i);

					byte[] buf = new byte[(long)tensor.ElementsSize * tensor.ElementsCount];
					fs.Seek((long)offs, SeekOrigin.Begin);
					int bytesRead = fs.Read(buf, 0, buf.Length);
					tensor.SetBackend(buf);
				}
			}

			//ggufContext.Free();

			return model;
		}

		private static SafeGGmlGraph MagikaGraph(MagikaModel model)
		{
			Hparams hparams = model.hparams;

			SafeGGmlContext ggmlContext = new SafeGGmlContext(IntPtr.Zero, NoAllocateMemory: true);
			SafeGGmlGraph graph = ggmlContext.NewGraph();

			SafeGGmlTensor input = ggmlContext.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 257, 1536, 1); // one-hot
			input.Name = "input";
			input.SetInput();

			SafeGGmlTensor cur;

			// dense
			cur = ggmlContext.MulMat(model.denseWeight, input);
			cur = ggmlContext.Add(cur, model.denseBias); // [128, 1536, n_files]
			cur = ggmlContext.Gelu(cur);

			// reshape
			cur = ggmlContext.Reshape3d(cur, 512, 384, 1); // [384, 512, n_files]
			cur = ggmlContext.Cont(ggmlContext.Transpose(cur));

			// layer normalization
			cur = ggmlContext.Norm(cur, hparams.normEps);
			cur = ggmlContext.Mul(cur, model.layerNormGamma); // [384, 512, n_files]
			cur = ggmlContext.Add(cur, model.layerNormBeta);  // [384, 512, n_files]

			// dense_1
			cur = ggmlContext.Cont(ggmlContext.Transpose(cur));
			cur = ggmlContext.MulMat(model.dense1Weight, cur);
			cur = ggmlContext.Add(cur, model.dense1Bias); // [256, 384, n_files]
			cur = ggmlContext.Gelu(cur);

			// dense_2
			cur = ggmlContext.MulMat(model.dense2Weight, cur);
			cur = ggmlContext.Add(cur, model.dense2Bias); // [256, 384, n_files]
			cur = ggmlContext.Gelu(cur);

			// global_max_pooling1d
			cur = ggmlContext.Cont(ggmlContext.Transpose(cur)); // [384, 256, n_files]
			cur = ggmlContext.Pool1d(cur, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 384, 384, 0); // [1, 256, n_files]
			cur = ggmlContext.Reshape2d(cur, 256, 1); // [256, n_files]

			// layer normalization 1
			cur = ggmlContext.Norm(cur, hparams.normEps);
			cur = ggmlContext.Mul(cur, model.layerNorm1Gamma); // [256, n_files]
			cur = ggmlContext.Add(cur, model.layerNorm1Beta);  // [256, n_files]

			// target_label
			cur = ggmlContext.MulMat(model.targetLabelWeight, cur);
			cur = ggmlContext.Add(cur, model.targetLabelBias); // [labelCount, n_files]
			cur = ggmlContext.SoftMax(cur); // [labelCount, n_files]
			cur.Name = "targetLabelProbs";
			cur.SetOutput();

			graph.BuildForwardExpend(cur);

			return graph;
		}

		private static float[] Eval(MagikaModel model, string fname)
		{
			Hparams hparams = model.hparams;
			SafeGGmlGraphAllocr alloc = new SafeGGmlGraphAllocr(model.backend.GetDefaultBufferType());

			SafeGGmlGraph graph = MagikaGraph(model);

			if (!graph.GraphAllocate(alloc))
			{
				throw new Exception("ggml_gallocr_alloc_graph() failed");
			}

			SafeGGmlTensor input = graph.GetTensor("input");

			var buf = new List<int>(Enumerable.Repeat(hparams.paddingToken, 1536));

			using (FileStream fileStream = new FileStream(fname, FileMode.Open, FileAccess.Read))
			{
				var fsize = fileStream.Length;
				long size = Math.Max(Math.Max(hparams.midSize, hparams.endSize), hparams.begSize);
				byte[] read_buf = new byte[size];

				// Read	beg
				int bytesToRead = fileStream.Read(read_buf, 0, hparams.begSize);
				for (int j = 0; j < bytesToRead; j++)
				{
					buf[j] = read_buf[j];
				}

				// Read mid
				var midOffs = Math.Max(0, (int)(fsize - hparams.midSize) / 2);
				fileStream.Seek(midOffs, SeekOrigin.Begin);
				bytesToRead = fileStream.Read(read_buf, 0, hparams.midSize);
				for (int j = 0; j < bytesToRead; j++)
				{
					// pad at both ends
					int mid_idx = hparams.begSize + (hparams.midSize / 2) - bytesToRead / 2 + j;
					buf[mid_idx] = read_buf[j];
				}

				// Read end

				var endOffs = Math.Max(0, fsize - hparams.endSize);
				fileStream.Seek(endOffs, SeekOrigin.Begin);
				bytesToRead = fileStream.Read(read_buf, 0, hparams.endSize);
				for (int j = 0; j < bytesToRead; j++)
				{
					// pad at the beginning
					int end_idx = hparams.begSize + hparams.midSize + hparams.endSize - bytesToRead + j;
					buf[end_idx] = read_buf[j];
				}
			}

			var inpBytes = hparams.begSize + hparams.midSize + hparams.endSize;
			var oneHot = new float[257 * inpBytes];
			for (int j = 0; j < inpBytes; j++)
			{
				oneHot[257 * j + buf[j]] = 1.0f;
			}

			input.SetBackend(oneHot);
			if (graph.BackendCompute(model.backend) != Structs.GGmlStatus.GGML_STATUS_SUCCESS)
			{
				throw new Exception("ggml_backend_graph_compute() failed");
			}

			SafeGGmlTensor targetLabelProbs = graph.GetTensor("targetLabelProbs");

			byte[] bytes = targetLabelProbs.GetBackend();
			float[] probs = DataConverter.ConvertToFloats(bytes);
			return probs;
		}

		struct result
		{
			public string label;
			public float score;
		}

		static void Main(string[] args)
		{
			MagikaModel model = LoadModel("./Assets/magika.gguf");
			Console.WriteLine("Loaded model");

			float[] result = Eval(model, "./Assets/test");
			List<result> results = new List<result>();
			for (int i = 0; i < result.Length; i++)
			{
				results.Add(new result { label = Labels[i], score = result[i] });
			}

			results.Sort((a, b) => b.score.CompareTo(a.score));
			for (int i = 0; i < 5; i++)
			{
				Console.WriteLine("{0}: {1}", results[i].label, results[i].score);
			}
			Console.ReadKey();
		}
	}
}
