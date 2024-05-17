using GGMLSharp;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace magika
{
	internal unsafe class Program
	{
		static string[] magika_labels ={
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

		private class magika_hparams
		{
			public int block_size = 4096;
			public int beg_size = 512;
			public int mid_size = 512;
			public int end_size = 512;
			public int min_file_size_for_dl = 16;
			public int n_label = 113;
			public float f_norm_eps = 0.001f;
			public int padding_token = 256;
		};

		private class magika_model
		{
			~magika_model()
			{
				Native.ggml_backend_buffer_free(buf_w);
				Native.ggml_backend_free(backend);
				Native.ggml_free(ctx_w);
			}

			public magika_hparams hparams = new magika_hparams();

			public ggml_tensor* dense_w;
			public ggml_tensor* dense_b;

			public ggml_tensor* layer_norm_gamma;
			public ggml_tensor* layer_norm_beta;

			public ggml_tensor* dense_1_w;
			public ggml_tensor* dense_1_b;

			public ggml_tensor* dense_2_w;
			public ggml_tensor* dense_2_b;

			public ggml_tensor* layer_norm_1_gamma;
			public ggml_tensor* layer_norm_1_beta;

			public ggml_tensor* target_label_w;
			public ggml_tensor* target_label_b;

			public ggml_backend* backend = Native.ggml_backend_cpu_init();
			public ggml_backend_buffer* buf_w = null;
			public ggml_context* ctx_w = null;
		};

		private static ggml_tensor* checked_get_tensor(ggml_context* ctx, string name)
		{
			ggml_tensor* tensor = Native.ggml_get_tensor(ctx, name);
			if (null == tensor)
			{
				throw new ArgumentNullException($"tensor {name} not found");
			}
			return tensor;
		}

		private static magika_model magika_model_load(string fname)
		{
			magika_model model = new magika_model();
			ggml_context* ctx = model.ctx_w;

			gguf_init_params @params = new gguf_init_params
			{
				no_alloc = true,
				ctx = &ctx,
			};

			gguf_context* ctx_gguf = Native.gguf_init_from_file(fname, @params);
			if (null == ctx_gguf)
			{
				throw new FileLoadException($"gguf_init_from_file() failed");
			}

			model.buf_w = Native.ggml_backend_alloc_ctx_tensors(ctx, model.backend);
			if (null == model.buf_w)
			{
				throw new Exception($"%s: ggml_backend_alloc_ctx_tensors() failed");
				//Native.gguf_free(ctx_gguf);
			}

			try
			{
				model.dense_w = checked_get_tensor(ctx, "dense/kernel:0");
				model.dense_b = checked_get_tensor(ctx, "dense/bias:0");

				model.layer_norm_gamma = checked_get_tensor(ctx, "layer_normalization/gamma:0");
				model.layer_norm_beta = checked_get_tensor(ctx, "layer_normalization/beta:0");

				model.dense_1_w = checked_get_tensor(ctx, "dense_1/kernel:0");
				model.dense_1_b = checked_get_tensor(ctx, "dense_1/bias:0");

				model.dense_2_w = checked_get_tensor(ctx, "dense_2/kernel:0");
				model.dense_2_b = checked_get_tensor(ctx, "dense_2/bias:0");

				model.layer_norm_1_gamma = checked_get_tensor(ctx, "layer_normalization_1/gamma:0");
				model.layer_norm_1_beta = checked_get_tensor(ctx, "layer_normalization_1/beta:0");

				model.target_label_w = checked_get_tensor(ctx, "target_label/kernel:0");
				model.target_label_b = checked_get_tensor(ctx, "target_label/bias:0");
			}
			catch (Exception ex)
			{
				Console.WriteLine(ex.Message);
				Native.gguf_free(ctx_gguf);
				return null;
			}

			using (FileStream fs = new FileStream(fname, FileMode.Open, FileAccess.Read))
			{
				int n_tensors = Native.gguf_get_n_tensors(ctx_gguf);

				for (int i = 0; i < n_tensors; i++)
				{
					string? name = Native.gguf_get_tensor_name(ctx_gguf, i);

					ggml_tensor* tensor = Native.ggml_get_tensor(ctx, name);
					long offs = Native.gguf_get_data_offset(ctx_gguf) + Native.gguf_get_tensor_offset(ctx_gguf, i);


					long n_bytes = Native.ggml_nbytes(tensor);
					byte[] buf = new byte[n_bytes];


					fs.Seek(offs, SeekOrigin.Begin);
					int bytesRead = fs.Read(buf, 0, buf.Length);
					IntPtr buf_data = Marshal.UnsafeAddrOfPinnedArrayElement(buf, 0);

					Native.ggml_backend_tensor_set(tensor, buf_data, 0, bytesRead);
				}
			}

			Native.gguf_free(ctx_gguf);

			return model;
		}

		private static ggml_cgraph* magika_graph(magika_model model)
		{
			int GGML_DEFAULT_GRAPH_SIZE = 2048;
			magika_hparams hparams = model.hparams;
			long buf_size = Native.ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + Native.ggml_graph_overhead();

			ggml_init_params @params = new ggml_init_params
			{
				mem_buffer = IntPtr.Zero,
				mem_size = buf_size,
				no_alloc = true,
			};

			ggml_context* ctx = Native.ggml_init(@params);
			ggml_cgraph* gf = Native.ggml_new_graph(ctx);

			ggml_tensor* input = Native.ggml_new_tensor_3d(ctx, ggml_type.GGML_TYPE_F32, 257, 1536, 1); // one-hot
			Native.ggml_set_name(input, "input");
			Native.ggml_set_input(input);

			ggml_tensor* cur;

			// dense
			cur = Native.ggml_mul_mat(ctx, model.dense_w, input);
			cur = Native.ggml_add(ctx, cur, model.dense_b); // [128, 1536, n_files]
			cur = Native.ggml_gelu(ctx, cur);

			// reshape
			cur = Native.ggml_reshape_3d(ctx, cur, 512, 384, 1); // [384, 512, n_files]
			cur = Native.ggml_cont(ctx, Native.ggml_transpose(ctx, cur));

			// layer normalization
			cur = Native.ggml_norm(ctx, cur, hparams.f_norm_eps);
			cur = Native.ggml_mul(ctx, cur, model.layer_norm_gamma); // [384, 512, n_files]
			cur = Native.ggml_add(ctx, cur, model.layer_norm_beta);  // [384, 512, n_files]

			// dense_1
			cur = Native.ggml_cont(ctx, Native.ggml_transpose(ctx, cur));
			cur = Native.ggml_mul_mat(ctx, model.dense_1_w, cur);
			cur = Native.ggml_add(ctx, cur, model.dense_1_b); // [256, 384, n_files]
			cur = Native.ggml_gelu(ctx, cur);

			// dense_2
			cur = Native.ggml_mul_mat(ctx, model.dense_2_w, cur);
			cur = Native.ggml_add(ctx, cur, model.dense_2_b); // [256, 384, n_files]
			cur = Native.ggml_gelu(ctx, cur);

			// global_max_pooling1d
			cur = Native.ggml_cont(ctx, Native.ggml_transpose(ctx, cur)); // [384, 256, n_files]
			cur = Native.ggml_pool_1d(ctx, cur, ggml_op_pool.GGML_OP_POOL_MAX, 384, 384, 0); // [1, 256, n_files]
			cur = Native.ggml_reshape_2d(ctx, cur, 256, 1); // [256, n_files]

			// layer normalization 1
			cur = Native.ggml_norm(ctx, cur, hparams.f_norm_eps);
			cur = Native.ggml_mul(ctx, cur, model.layer_norm_1_gamma); // [256, n_files]
			cur = Native.ggml_add(ctx, cur, model.layer_norm_1_beta);  // [256, n_files]

			// target_label
			cur = Native.ggml_mul_mat(ctx, model.target_label_w, cur);
			cur = Native.ggml_add(ctx, cur, model.target_label_b); // [n_label, n_files]
			cur = Native.ggml_soft_max(ctx, cur); // [n_label, n_files]
			Native.ggml_set_name(cur, "target_label_probs");
			Native.ggml_set_output(cur);

			Native.ggml_build_forward_expand(gf, cur);

			return gf;
		}

		private static float[] magika_eval(magika_model model, string fname)
		{
			magika_hparams hparams = model.hparams;
			ggml_gallocr* alloc = Native.ggml_gallocr_new(Native.ggml_backend_get_default_buffer_type(model.backend));

			ggml_cgraph* gf = magika_graph(model);

			if (!Native.ggml_gallocr_alloc_graph(alloc, gf))
			{
				throw new Exception("ggml_gallocr_alloc_graph() failed");
			}

			ggml_tensor* input = Native.ggml_graph_get_tensor(gf, "input");

			var buf = new List<int>(Enumerable.Repeat(hparams.padding_token, 1536));

			using (FileStream fileStream = new FileStream(fname, FileMode.Open, FileAccess.Read))
			{
				var fsize = fileStream.Length;
				long size = Math.Max(Math.Max(hparams.mid_size, hparams.end_size), hparams.beg_size);
				byte[] read_buf = new byte[size];

				// Read	beg
				int n_read = fileStream.Read(read_buf, 0, hparams.beg_size);
				for (int j = 0; j < n_read; j++)
				{
					buf[j] = read_buf[j];
				}

				// Read mid
				var midOffs = Math.Max(0, (int)(fsize - hparams.mid_size) / 2);
				fileStream.Seek(midOffs, SeekOrigin.Begin);
				n_read = fileStream.Read(read_buf, 0, hparams.mid_size);
				for (int j = 0; j < n_read; j++)
				{
					// pad at both ends
					int mid_idx = hparams.beg_size + (hparams.mid_size / 2) - n_read / 2 + j;
					buf[mid_idx] = read_buf[j];
				}

				// Read end

				var endOffs = Math.Max(0, fsize - hparams.end_size);
				fileStream.Seek(endOffs, SeekOrigin.Begin);
				n_read = fileStream.Read(read_buf, 0, hparams.end_size);
				for (int j = 0; j < n_read; j++)
				{
					// pad at the beginning
					int end_idx = hparams.beg_size + hparams.mid_size + hparams.end_size - n_read + j;
					buf[end_idx] = read_buf[j];
				}
			}

			var inpBytes = hparams.beg_size + hparams.mid_size + hparams.end_size;
			var oneHot = new float[257 * inpBytes];
			for (int j = 0; j < inpBytes; j++)
			{
				oneHot[257 * j + buf[j]] = 1.0f;
			}

			Native.ggml_backend_tensor_set(input, Marshal.UnsafeAddrOfPinnedArrayElement(oneHot, 0), 0, 257 * inpBytes * sizeof(float));
			if (Native.ggml_backend_graph_compute(model.backend, gf) != ggml_status.GGML_STATUS_SUCCESS)
			{
				throw new Exception("ggml_backend_graph_compute() failed");
			}

			ggml_tensor* target_label_probs = Native.ggml_graph_get_tensor(gf, "target_label_probs");

			float[] probs = new float[hparams.n_label];
			Native.ggml_backend_tensor_get(target_label_probs, Marshal.UnsafeAddrOfPinnedArrayElement(probs, 0), 0, hparams.n_label * sizeof(float));

			return probs;
		}


		static void Main(string[] args)
		{
			magika_model model = magika_model_load(@".\Assets\magika.gguf");
			float[] result_tensor = magika_eval(model, @".\Assets\test");
			List<result> results = new List<result>();
			for (int i = 0; i < result_tensor.Length; i++)
			{
				results.Add(new result { label = magika_labels[i], score = result_tensor[i] });
			}

			results.Sort((a, b) => b.score.CompareTo(a.score));
			for (int i = 0; i < 5; i++)
			{
				Console.WriteLine("{0}: {1}", results[i].label, results[i].score);
			}
		}

		class result
		{
			public string label;
			public float score;
		}
	}
}
