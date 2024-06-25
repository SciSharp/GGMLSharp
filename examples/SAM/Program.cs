using GGMLSharp;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace SAM
{
	internal unsafe class Program
	{
		static void Main(string[] args)
		{
			// First you shold download the sam_vit_b model and move it to Assets folder.
			// The download link is: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

			sam_params sam_param = new sam_params()
			{
				iou_threshold = 0.8f,
				stability_score_threshold = 0.9f,
				pt = new sam_point() { x = 248.0f, y = 162.0f, },
				model = @"./Assets/sam_vit_b_01ec64.pth",
				fname_inp = @"./Assets/example.jpg",
			};
			// Load model
			sam_model model = sam_model_load(sam_param);
			Console.WriteLine("Model Loaded.");

			Console.WriteLine("Load image from file...");
			// Load Img
			sam_image_u8 img_u8 = sam_image_load_from_file(sam_param.fname_inp);
			sam_image_f32 img_f32 = sam_image_preprocess(img_u8);

			Console.WriteLine("Init state");
			// Init state
			sam_state state = new sam_state();
			{
				ulong buf_size = 256u * 1024 * 1024;
				ggml_init_params ggml_params = new ggml_init_params()
				{
					mem_size = buf_size,
					mem_buffer = IntPtr.Zero,
					no_alloc = false
				};

				state.ctx = Native.ggml_init(ggml_params);
				state.embd_img = Native.ggml_new_tensor_3d(state.ctx, ggml_type.GGML_TYPE_F32, model.hparams.n_img_embd(), model.hparams.n_img_embd(), model.hparams.n_enc_out_chans);
				state.low_res_masks = Native.ggml_new_tensor_3d(state.ctx, ggml_type.GGML_TYPE_F32, model.hparams.n_enc_out_chans, model.hparams.n_enc_out_chans, 3);
				state.iou_predictions = Native.ggml_new_tensor_1d(state.ctx, ggml_type.GGML_TYPE_F32, 3);
			}

			// Encode image
			{
				Console.WriteLine("Encoding the image...");
				state.buf_compute_img_enc = new byte[Native.ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + Native.ggml_graph_overhead()];
				state.allocr = Native.ggml_gallocr_new(Native.ggml_backend_cpu_buffer_type());

				ggml_cgraph* gf = sam_encode_image(model, state, img_f32);
				if (null == gf)
				{
					throw new Exception("failed to encode image");
				}

				ggml_graph_compute_helper(state.work_buffer, gf, sam_param.n_threads);

				Native.ggml_gallocr_free(state.allocr);
				state.allocr = null;
				Console.WriteLine("Image encoded done.");
			}

			// Compute the masks
			{
				state.allocr = Native.ggml_gallocr_new(Native.ggml_backend_cpu_buffer_type());

				// TODO: more varied prompts
				Console.WriteLine($"prompt point: ({sam_param.pt.x}, {sam_param.pt.y})");
				Console.WriteLine("Computing the mask...");
				ggml_cgraph* gf = sam_build_fast_graph(model, state, img_u8.nx, img_u8.ny, sam_param.pt);

				if (null == gf)
				{
					throw new Exception("failed to build fast graph");
				}

				ggml_graph_compute_helper(state.work_buffer, gf, sam_param.n_threads);

				Native.ggml_gallocr_free(state.allocr);
				state.allocr = null;
				Console.WriteLine("Masks have got.");
			}


			// Write masks.

			Console.WriteLine("Writing masks...");
			sam_write_masks(model.hparams, img_u8.nx, img_u8.ny, state, sam_param.fname_out);

			Console.WriteLine("Write masks done.");
		}

		class sam_hparams
		{
			public int n_enc_state = 768;
			public int n_enc_layer = 12;
			public int n_enc_head = 12;
			public int n_enc_out_chans = 256;
			public int n_pt_embd = 4;
			public int n_dec_heads = 8;
			public int ftype = 1;
			public float mask_threshold = 0.0f;
			public float iou_threshold = 0.88f;
			public float stability_score_threshold = 0.95f;
			public float stability_score_offset = 1.0f;
			public float eps = 1e-6f;
			public float eps_decoder_transformer = 1e-5f;

			public int n_enc_head_dim() { return n_enc_state / n_enc_head; }
			public int n_img_size()
			{
				return 1024;
			}
			public int n_window_size() { return 14; }
			public int n_patch_size() { return 16; }
			public int n_img_embd()
			{
				return n_img_size() / n_patch_size();
			}

			public int[] global_attn_indices()
			{
				switch (n_enc_state)
				{
					case 768: return [2, 5, 8, 11];
					case 1024: return [5, 11, 17, 23];
					case 1280: return [7, 15, 23, 31];
					default:
						{
							throw new ArgumentOutOfRangeException($"unsupported n_enc_state = {n_enc_state}");
						}
				};
			}

			public bool is_global_attn(int layer)
			{
				return global_attn_indices().Contains(layer);
			}
		};

		struct sam_point
		{
			public float x;
			public float y;
		};

		// RGB uint8 image
		struct sam_image_u8
		{
			public int nx;
			public int ny;

			public byte[] data;
		};

		// RGB float32 image
		// Memory layout: RGBRGBRGB...
		struct sam_image_f32
		{
			public int nx;
			public int ny;

			public float[] data;
		};

		class sam_model
		{
			public sam_hparams hparams = new sam_hparams();

			public sam_encoder_image enc_img;
			public sam_encoder_prompt enc_prompt;
			public sam_decoder_mask dec;

			public ggml_context* ctx;

			//public List<named_tensor> tensors;
		};

		struct named_tensor
		{
			public string name;
			public ggml_tensor* tensor;
		};

		struct sam_layer_enc
		{
			public ggml_tensor* norm1_w;
			public ggml_tensor* norm1_b;

			public ggml_tensor* rel_pos_w;
			public ggml_tensor* rel_pos_h;

			public ggml_tensor* qkv_w;
			public ggml_tensor* qkv_b;

			public ggml_tensor* proj_w;
			public ggml_tensor* proj_b;

			public ggml_tensor* norm2_w;
			public ggml_tensor* norm2_b;

			public ggml_tensor* mlp_lin1_w;
			public ggml_tensor* mlp_lin1_b;

			public ggml_tensor* mlp_lin2_w;
			public ggml_tensor* mlp_lin2_b;
		};

		struct sam_encoder_image
		{
			public ggml_tensor* pe;

			public ggml_tensor* proj_w;
			public ggml_tensor* proj_b;

			public ggml_tensor* neck_conv_0;
			public ggml_tensor* neck_norm_0_w;
			public ggml_tensor* neck_norm_0_b;
			public ggml_tensor* neck_conv_1;
			public ggml_tensor* neck_norm_1_w;
			public ggml_tensor* neck_norm_1_b;

			public sam_layer_enc[] layers;
		};

		struct sam_encoder_prompt
		{
			public ggml_tensor* pe;

			public ggml_tensor* not_a_pt_embd_w;
			public ggml_tensor** pt_embd;

			public ggml_tensor* no_mask_embd_w;
			//std::vector<  ggml_tensor *> mask_down_w;
			//std::vector<  ggml_tensor *> mask_down_b;
		};

		struct sam_layer_dec_transformer_attn
		{
			// q_proj
			public ggml_tensor* q_w;
			public ggml_tensor* q_b;

			// k_proj
			public ggml_tensor* k_w;
			public ggml_tensor* k_b;

			// v_proj
			public ggml_tensor* v_w;
			public ggml_tensor* v_b;

			// out_proj
			public ggml_tensor* out_w;
			public ggml_tensor* out_b;
		};

		struct sam_layer_dec_transformer
		{
			public sam_layer_dec_transformer_attn self_attn;

			// norm1
			public ggml_tensor* norm1_w;
			public ggml_tensor* norm1_b;

			public sam_layer_dec_transformer_attn cross_attn_token_to_img;

			// norm2
			public ggml_tensor* norm2_w;
			public ggml_tensor* norm2_b;

			// mlp.lin1
			public ggml_tensor* mlp_lin1_w;
			public ggml_tensor* mlp_lin1_b;

			// mlp.lin2
			public ggml_tensor* mlp_lin2_w;
			public ggml_tensor* mlp_lin2_b;

			// norm3
			public ggml_tensor* norm3_w;
			public ggml_tensor* norm3_b;

			// norm4
			public ggml_tensor* norm4_w;
			public ggml_tensor* norm4_b;

			public sam_layer_dec_transformer_attn cross_attn_img_to_token;
		};

		struct sam_layer_dec_output_hypernet_mlps
		{
			// mlps_*.layers.0
			public ggml_tensor* w_0;
			public ggml_tensor* b_0;

			// mlps_*.layers.1
			public ggml_tensor* w_1;
			public ggml_tensor* b_1;

			// mlps_*.layers.2
			public ggml_tensor* w_2;
			public ggml_tensor* b_2;
		};

		struct sam_decoder_mask
		{
			public sam_layer_dec_transformer[] transformer_layers;

			// trasnformer.final_attn_token_to_image
			public sam_layer_dec_transformer_attn transformer_final_attn_token_to_img;

			// transformer.norm_final
			public ggml_tensor* transformer_norm_final_w;
			public ggml_tensor* transformer_norm_final_b;

			// output_upscaling.0
			public ggml_tensor* output_upscaling_0_w;
			public ggml_tensor* output_upscaling_0_b;

			// output_upscaling.1
			public ggml_tensor* output_upscaling_1_w;
			public ggml_tensor* output_upscaling_1_b;

			// output_upscaling.3
			public ggml_tensor* output_upscaling_3_w;
			public ggml_tensor* output_upscaling_3_b;

			// output_hypernetworks_mlps
			public sam_layer_dec_output_hypernet_mlps[] output_hypernet_mlps;

			// iou_prediction_head.0
			public ggml_tensor* iou_prediction_head_0_w;
			public ggml_tensor* iou_prediction_head_0_b;

			// iou_prediction_head.1
			public ggml_tensor* iou_prediction_head_1_w;
			public ggml_tensor* iou_prediction_head_1_b;

			// iou_prediction_head.2
			public ggml_tensor* iou_prediction_head_2_w;
			public ggml_tensor* iou_prediction_head_2_b;

			// iou_token.weight
			public ggml_tensor* iou_token_w;

			// mask_tokens.weight
			public ggml_tensor* mask_tokens_w;
		};

		struct sam_state
		{
			public ggml_tensor* embd_img;

			public ggml_tensor* low_res_masks;
			public ggml_tensor* iou_predictions;

			//  ggml_tensor * tmp_save = {};

			public ggml_context* ctx;

			// buffer for `ggml_graph_plan.work_data`
			public byte[] work_buffer;
			// buffers to evaluate the model
			public byte[] buf_compute_img_enc;

			public byte[] buf_compute_fast;

			public ggml_gallocr* allocr;
		};

		class sam_params
		{
			public int seed = -1; // RNG seed
			public int n_threads = 4;

			public string model = "./Assets/sam_vit_b_01ec64.pth"; // model path
			public string fname_inp = "./Assets/example.jpg";
			public string fname_out = "img.out";
			public float mask_threshold = 0.0f;
			public float iou_threshold = 0.88f;
			public float stability_score_threshold = 0.95f;
			public float stability_score_offset = 1.0f;
			public float eps = 1e-6f;
			public float eps_decoder_transformer = 1e-5f;
			public sam_point pt = new sam_point() { x = 414.375f, y = 162.796875f, };
		};

		static void ggml_disconnect_node_from_graph(ggml_tensor* t)
		{
			t->op = ggml_op.GGML_OP_NONE;
			for (int i = 0; i < Structs.GGML_MAX_SRC; i++)
			{
				t->src[i] = 0;
			}
		}

		static void ggml_graph_compute_helper(byte[] buf, ggml_cgraph* graph, int n_threads)
		{
			ggml_cplan plan = Native.ggml_graph_plan(graph, n_threads);

			if (plan.work_size > 0)
			{
				//Array.Resize(ref buf, (int)plan.work_size);
				plan.work_data = (byte*)Marshal.AllocHGlobal((int)plan.work_size);
			}


			ggml_init_params @params = new ggml_init_params()
			{
				mem_buffer = IntPtr.Zero,
				mem_size = Native.ggml_tensor_overhead() * Structs.GGML_DEFAULT_GRAPH_SIZE * (ulong)graph->n_nodes + Native.ggml_graph_overhead(),
				no_alloc = false,
			};
			ggml_context* ctx = Native.ggml_init(@params);
			Native.ggml_graph_compute_with_ctx(ctx, graph, n_threads);
			//Native.ggml_graph_compute(graph, &plan);
			Native.ggml_free(ctx);
		}

		static void ggml_sam_sin(ggml_tensor* dst, ggml_tensor* src, int ith, int nth, void* userdata)
		{
			if (userdata != null)
			{
				throw new Exception("userdata is null");
			}
			if (!Native.ggml_are_same_shape(dst, src))
			{
				throw new Exception("dst and src are not the same shape");
			}
			if (!Native.ggml_is_contiguous(dst))
			{
				throw new Exception("des is not contiguous");
			}
			if (!Native.ggml_is_contiguous(src))
			{
				throw new Exception("src is not contiguous");
			}

			float* src_data = Native.ggml_get_data_f32(src);
			float* dst_data = Native.ggml_get_data_f32(dst);

			int ne = (int)Native.ggml_nelements(dst);
			int dr = (ne + nth - 1) / nth;
			int ie0 = dr * ith;
			int ie1 = Math.Min(ie0 + dr, ne);

			for (int i = ie0; i < ie1; ++i)
			{
				dst_data[i] = MathF.Sin(src_data[i]);
			}
		}

		static void ggml_sam_cos(ggml_tensor* dst, ggml_tensor* src, int ith, int nth, void* userdata)
		{
			if (userdata != null)
			{
				throw new Exception("userdata is null");
			}
			if (!Native.ggml_are_same_shape(dst, src))
			{
				throw new Exception("dst and src are not the same shape");
			}
			if (!Native.ggml_is_contiguous(dst))
			{
				throw new Exception("des is not contiguous");
			}
			if (!Native.ggml_is_contiguous(src))
			{
				throw new Exception("src is not contiguous");
			}

			float* src_data = Native.ggml_get_data_f32(src);
			float* dst_data = Native.ggml_get_data_f32(dst);

			int ne = (int)Native.ggml_nelements(dst);
			int dr = (ne + nth - 1) / nth;
			int ie0 = dr * ith;
			int ie1 = Math.Min(ie0 + dr, ne);

			for (int i = ie0; i < ie1; ++i)
			{
				dst_data[i] = MathF.Cos(src_data[i]);
			}
		}

		static sam_image_u8 sam_image_load_from_file(string fname)
		{
			Bitmap bitmap = new Bitmap(fname);
			sam_image_u8 img = new sam_image_u8();
			img.nx = bitmap.Width;
			img.ny = bitmap.Height;
			img.data = new byte[img.nx * img.ny * 3];

			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
			Marshal.Copy(bitmapData.Scan0, img.data, 0, img.data.Length);
			bitmap.UnlockBits(bitmapData);
			return img;
		}

		// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L164
		// resize largest dimension to 1024
		// normalize: x = (x - mean) / std
		//     mean = [123.675, 116.28, 103.53]
		//     std  = [58.395, 57.12, 57.375]
		//     TODO: why are these hardcoded !?
		// pad to 1024x1024
		// TODO: for some reason, this is not numerically identical to pytorch's interpolation
		static sam_image_f32 sam_image_preprocess(sam_image_u8 img)
		{
			sam_image_f32 res = new sam_image_f32();
			int nx = img.nx;
			int ny = img.ny;

			int nx2 = 1024;
			int ny2 = 1024;

			res.nx = nx2;
			res.ny = ny2;
			res.data = new float[3 * nx2 * ny2];

			float scale = MathF.Max(nx, ny) / 1024.0f;

			Console.WriteLine($"scale = {scale}");

			int nx3 = (int)(nx / scale + 0.5f);
			int ny3 = (int)(ny / scale + 0.5f);

			float[] m3 = { 123.675f, 116.280f, 103.530f };
			float[] s3 = { 58.395f, 57.120f, 57.375f };

			for (int y = 0; y < ny3; y++)
			{
				for (int x = 0; x < nx3; x++)
				{
					for (int c = 0; c < 3; c++)
					{
						// linear interpolation
						float sx = (x + 0.5f) * scale - 0.5f;
						float sy = (y + 0.5f) * scale - 0.5f;

						int x0 = Math.Max(0, (int)Math.Floor(sx));
						int y0 = Math.Max(0, (int)Math.Floor(sy));

						int x1 = Math.Min(x0 + 1, nx - 1);
						int y1 = Math.Min(y0 + 1, ny - 1);

						float dx = sx - x0;
						float dy = sy - y0;

						int j00 = 3 * (y0 * nx + x0) + c;
						int j01 = 3 * (y0 * nx + x1) + c;
						int j10 = 3 * (y1 * nx + x0) + c;
						int j11 = 3 * (y1 * nx + x1) + c;

						float v00 = img.data[j00];
						float v01 = img.data[j01];
						float v10 = img.data[j10];
						float v11 = img.data[j11];

						float v0 = v00 * (1.0f - dx) + v01 * dx;
						float v1 = v10 * (1.0f - dx) + v11 * dx;

						float v = v0 * (1.0f - dy) + v1 * dy;

						double v2 = Math.Min(Math.Max(Math.Round(v), 0.0f), 255.0f);

						int i = 3 * (y * nx3 + x) + c;

						res.data[i] = ((float)(v2) - m3[c]) / s3[c];
					}
				}
			}

			return res;
		}

		static sam_model sam_model_load(sam_params sam_params)
		{
			if (!File.Exists(sam_params.model))
			{
				throw new FileNotFoundException("Model file not found");
			}

			Console.WriteLine($"loading model from {sam_params.model} - please wait ...");
			List<PickleLoader.CommonTensor> tensors = PickleLoader.ReadTensorInfoFromFile(sam_params.model);
			Console.WriteLine($"model header loaded, total layers is: {tensors.Count}");


			ulong ctx_size = 0;
			tensors.ForEach(a =>
			{
				ulong tensorSize = Native.ggml_type_size(a.Type);
				a.Shape.ForEach(ne =>
				{
					tensorSize *= ne;
				});
				ctx_size += tensorSize;
			});
			ctx_size += Native.ggml_tensor_overhead() * (ulong)tensors.Count;
			ggml_init_params ggml_params = new ggml_init_params
			{
				mem_size = ctx_size,
				mem_buffer = IntPtr.Zero,
				no_alloc = false,
			};

			ggml_context* ctx = Native.ggml_init(ggml_params);

			sam_model model = new sam_model();
			model.ctx = ctx;

			//sam_hparams hparams = model.hparams;
			sam_hparams hparams = new sam_hparams
			{
				eps = sam_params.eps,
				eps_decoder_transformer = sam_params.eps_decoder_transformer,
				iou_threshold = sam_params.iou_threshold,
				mask_threshold = sam_params.mask_threshold,
				stability_score_offset = sam_params.stability_score_offset,
				stability_score_threshold = sam_params.stability_score_threshold,
			};

			model.hparams = hparams;

			int n_enc_state = hparams.n_enc_state;
			int n_enc_layer = hparams.n_enc_layer;
			int n_enc_head_dim = hparams.n_enc_head_dim();
			int n_enc_out_chans = hparams.n_enc_out_chans;
			int n_pt_embd = hparams.n_pt_embd;

			int n_img_embd = hparams.n_img_embd();
			int n_window_size = hparams.n_window_size();
			int n_patch_size = hparams.n_patch_size();


			// image encoder
			{
				model.enc_img.pe = GetTensorFromName(ctx, tensors, "image_encoder.pos_embed", [n_enc_state, n_img_embd, n_img_embd]);

				model.enc_img.proj_w = GetTensorFromName(ctx, tensors, "image_encoder.patch_embed.proj.weight", [n_patch_size, n_patch_size, 3, n_enc_state], ggml_type.GGML_TYPE_F16);
				model.enc_img.proj_b = GetTensorFromName(ctx, tensors, "image_encoder.patch_embed.proj.bias", [1, 1, n_enc_state]);

				model.enc_img.neck_conv_0 = GetTensorFromName(ctx, tensors, "image_encoder.neck.0.weight", [1, 1, n_enc_state, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
				model.enc_img.neck_conv_1 = GetTensorFromName(ctx, tensors, "image_encoder.neck.2.weight", [3, 3, n_enc_out_chans, n_enc_out_chans], ggml_type.GGML_TYPE_F16);

				model.enc_img.neck_norm_0_w = GetTensorFromName(ctx, tensors, "image_encoder.neck.1.weight");
				model.enc_img.neck_norm_0_b = GetTensorFromName(ctx, tensors, "image_encoder.neck.1.bias");

				model.enc_img.neck_norm_1_w = GetTensorFromName(ctx, tensors, "image_encoder.neck.3.weight");
				model.enc_img.neck_norm_1_b = GetTensorFromName(ctx, tensors, "image_encoder.neck.3.bias");

				model.enc_img.layers = new sam_layer_enc[model.hparams.n_enc_layer];

				for (int i = 0; i < model.hparams.n_enc_layer; ++i)
				{
					model.enc_img.layers[i].norm1_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".norm1.weight");
					model.enc_img.layers[i].norm1_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".norm1.bias");
					if (hparams.is_global_attn(i))
					{
						model.enc_img.layers[i].rel_pos_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_w", [n_enc_head_dim, 2 * n_img_embd - 1], ggml_type.GGML_TYPE_F16);
						model.enc_img.layers[i].rel_pos_h = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_h", [n_enc_head_dim, 2 * n_img_embd - 1], ggml_type.GGML_TYPE_F16);
					}
					else
					{
						model.enc_img.layers[i].rel_pos_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_w", [n_enc_head_dim, 2 * n_window_size - 1], ggml_type.GGML_TYPE_F16);
						model.enc_img.layers[i].rel_pos_h = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_h", [n_enc_head_dim, 2 * n_window_size - 1], ggml_type.GGML_TYPE_F16);
					}

					model.enc_img.layers[i].qkv_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.qkv.weight", [n_enc_state, 3 * n_enc_state, 1, 1], ggml_type.GGML_TYPE_F16);
					model.enc_img.layers[i].qkv_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.qkv.bias");
					model.enc_img.layers[i].proj_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.proj.weight");
					model.enc_img.layers[i].proj_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".attn.proj.bias");
					model.enc_img.layers[i].norm2_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".norm2.weight");
					model.enc_img.layers[i].norm2_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".norm2.bias");
					model.enc_img.layers[i].mlp_lin1_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".mlp.lin1.weight", [n_enc_state, 4 * n_enc_state], ggml_type.GGML_TYPE_F16);
					model.enc_img.layers[i].mlp_lin1_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".mlp.lin1.bias");
					model.enc_img.layers[i].mlp_lin2_w = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".mlp.lin2.weight", [4 * n_enc_state, n_enc_state], ggml_type.GGML_TYPE_F16);
					model.enc_img.layers[i].mlp_lin2_b = GetTensorFromName(ctx, tensors, "image_encoder.blocks." + i + ".mlp.lin2.bias");
				}
			}

			// prompt encoder
			{
				model.enc_prompt.pe = GetTensorFromName(ctx, tensors, "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", [n_enc_out_chans / 2, 2], ggml_type.GGML_TYPE_F32);
				model.enc_prompt.not_a_pt_embd_w = GetTensorFromName(ctx, tensors, "prompt_encoder.not_a_point_embed.weight", [n_enc_out_chans, 1]);
				model.enc_prompt.no_mask_embd_w = GetTensorFromName(ctx, tensors, "prompt_encoder.no_mask_embed.weight", [n_enc_out_chans, 1]);
				model.enc_prompt.pt_embd = (ggml_tensor**)Marshal.AllocHGlobal(sizeof(ggml_tensor) * model.hparams.n_pt_embd);
				for (int i = 0; i < model.hparams.n_pt_embd; i++)
				{
					model.enc_prompt.pt_embd[i] = GetTensorFromName(ctx, tensors, "prompt_encoder.point_embeddings." + i + ".weight", [n_enc_out_chans, 1]);
				}
			}

			// mask decoder
			{
				int tfm_layers_count = 2;
				model.dec.transformer_layers = new sam_layer_dec_transformer[tfm_layers_count];
				for (int i = 0; i < tfm_layers_count; ++i)
				{
					string prefix = "mask_decoder.transformer.layers." + i + ".";
					model.dec.transformer_layers[i].self_attn.q_w = GetTensorFromName(ctx, tensors, prefix + "self_attn.q_proj.weight");
					model.dec.transformer_layers[i].self_attn.q_b = GetTensorFromName(ctx, tensors, prefix + "self_attn.q_proj.bias");
					model.dec.transformer_layers[i].self_attn.k_w = GetTensorFromName(ctx, tensors, prefix + "self_attn.k_proj.weight");
					model.dec.transformer_layers[i].self_attn.k_b = GetTensorFromName(ctx, tensors, prefix + "self_attn.k_proj.bias");
					model.dec.transformer_layers[i].self_attn.v_w = GetTensorFromName(ctx, tensors, prefix + "self_attn.v_proj.weight");
					model.dec.transformer_layers[i].self_attn.v_b = GetTensorFromName(ctx, tensors, prefix + "self_attn.v_proj.bias");
					model.dec.transformer_layers[i].self_attn.out_w = GetTensorFromName(ctx, tensors, prefix + "self_attn.out_proj.weight");
					model.dec.transformer_layers[i].self_attn.out_b = GetTensorFromName(ctx, tensors, prefix + "self_attn.out_proj.bias");
					model.dec.transformer_layers[i].norm1_w = GetTensorFromName(ctx, tensors, prefix + "norm1.weight");
					model.dec.transformer_layers[i].norm1_b = GetTensorFromName(ctx, tensors, prefix + "norm1.bias");
					model.dec.transformer_layers[i].cross_attn_token_to_img.q_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.q_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_token_to_img.q_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.q_proj.bias");
					model.dec.transformer_layers[i].cross_attn_token_to_img.k_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.k_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_token_to_img.k_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.k_proj.bias");
					model.dec.transformer_layers[i].cross_attn_token_to_img.v_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.v_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_token_to_img.v_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.v_proj.bias");
					model.dec.transformer_layers[i].cross_attn_token_to_img.out_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.out_proj.weight", [n_enc_out_chans / 2, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_token_to_img.out_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_token_to_image.out_proj.bias");
					model.dec.transformer_layers[i].norm2_w = GetTensorFromName(ctx, tensors, prefix + "norm2.weight");
					model.dec.transformer_layers[i].norm2_b = GetTensorFromName(ctx, tensors, prefix + "norm2.bias");
					model.dec.transformer_layers[i].mlp_lin1_w = GetTensorFromName(ctx, tensors, prefix + "mlp.lin1.weight", [n_enc_out_chans, 8 * n_enc_out_chans], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].mlp_lin1_b = GetTensorFromName(ctx, tensors, prefix + "mlp.lin1.bias");
					model.dec.transformer_layers[i].mlp_lin2_w = GetTensorFromName(ctx, tensors, prefix + "mlp.lin2.weight", [8 * n_enc_out_chans, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].mlp_lin2_b = GetTensorFromName(ctx, tensors, prefix + "mlp.lin2.bias");
					model.dec.transformer_layers[i].norm3_w = GetTensorFromName(ctx, tensors, prefix + "norm3.weight");
					model.dec.transformer_layers[i].norm3_b = GetTensorFromName(ctx, tensors, prefix + "norm3.bias");
					model.dec.transformer_layers[i].norm4_w = GetTensorFromName(ctx, tensors, prefix + "norm4.weight");
					model.dec.transformer_layers[i].norm4_b = GetTensorFromName(ctx, tensors, prefix + "norm4.bias");
					model.dec.transformer_layers[i].cross_attn_img_to_token.q_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.q_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_img_to_token.q_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.q_proj.bias");
					model.dec.transformer_layers[i].cross_attn_img_to_token.k_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.k_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_img_to_token.k_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.k_proj.bias");
					model.dec.transformer_layers[i].cross_attn_img_to_token.v_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.v_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_img_to_token.v_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.v_proj.bias");
					model.dec.transformer_layers[i].cross_attn_img_to_token.out_w = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.out_proj.weight", [n_enc_out_chans / 2, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
					model.dec.transformer_layers[i].cross_attn_img_to_token.out_b = GetTensorFromName(ctx, tensors, prefix + "cross_attn_image_to_token.out_proj.bias");
				}

				model.dec.transformer_final_attn_token_to_img.q_w = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.q_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
				model.dec.transformer_final_attn_token_to_img.q_b = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.q_proj.bias");
				model.dec.transformer_final_attn_token_to_img.k_w = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.k_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
				model.dec.transformer_final_attn_token_to_img.k_b = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.k_proj.bias");
				model.dec.transformer_final_attn_token_to_img.v_w = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.v_proj.weight", [n_enc_out_chans, n_enc_out_chans / 2], ggml_type.GGML_TYPE_F16);
				model.dec.transformer_final_attn_token_to_img.v_b = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.v_proj.bias");
				model.dec.transformer_final_attn_token_to_img.out_w = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.out_proj.weight", [n_enc_out_chans / 2, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
				model.dec.transformer_final_attn_token_to_img.out_b = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.final_attn_token_to_image.out_proj.bias");

				model.dec.transformer_norm_final_w = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.norm_final_attn.weight");
				model.dec.transformer_norm_final_b = GetTensorFromName(ctx, tensors, "mask_decoder.transformer.norm_final_attn.bias");

				model.dec.output_upscaling_0_w = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.0.weight", [2, 2, n_img_embd, n_enc_out_chans], ggml_type.GGML_TYPE_F16);
				model.dec.output_upscaling_0_b = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.0.bias");
				model.dec.output_upscaling_1_w = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.1.weight");
				model.dec.output_upscaling_1_b = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.1.bias");
				model.dec.output_upscaling_3_w = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.3.weight", [2, 2, n_img_embd / 2, n_img_embd], ggml_type.GGML_TYPE_F16);
				model.dec.output_upscaling_3_b = GetTensorFromName(ctx, tensors, "mask_decoder.output_upscaling.3.bias");

				int n_hypernet_mpls_count = 4;
				model.dec.output_hypernet_mlps = new sam_layer_dec_output_hypernet_mlps[n_hypernet_mpls_count];
				for (int i = 0; i < n_hypernet_mpls_count; ++i)
				{
					string prefix = "mask_decoder.output_hypernetworks_mlps." + i + ".";
					model.dec.output_hypernet_mlps[i].w_0 = GetTensorFromName(ctx, tensors, prefix + "layers.0.weight");
					model.dec.output_hypernet_mlps[i].b_0 = GetTensorFromName(ctx, tensors, prefix + "layers.0.bias");
					model.dec.output_hypernet_mlps[i].w_1 = GetTensorFromName(ctx, tensors, prefix + "layers.1.weight");
					model.dec.output_hypernet_mlps[i].b_1 = GetTensorFromName(ctx, tensors, prefix + "layers.1.bias");
					model.dec.output_hypernet_mlps[i].w_2 = GetTensorFromName(ctx, tensors, prefix + "layers.2.weight", [n_enc_out_chans, n_img_embd / 2]);
					model.dec.output_hypernet_mlps[i].b_2 = GetTensorFromName(ctx, tensors, prefix + "layers.2.bias");
				}

				model.dec.iou_prediction_head_0_w = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.0.weight");
				model.dec.iou_prediction_head_0_b = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.0.bias");
				model.dec.iou_prediction_head_1_w = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.1.weight");
				model.dec.iou_prediction_head_1_b = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.1.bias");
				model.dec.iou_prediction_head_2_w = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.2.weight", [n_enc_out_chans, n_pt_embd]);
				model.dec.iou_prediction_head_2_b = GetTensorFromName(ctx, tensors, "mask_decoder.iou_prediction_head.layers.2.bias");

				model.dec.iou_token_w = GetTensorFromName(ctx, tensors, "mask_decoder.iou_token.weight", [n_enc_out_chans, 1]);
				model.dec.mask_tokens_w = GetTensorFromName(ctx, tensors, "mask_decoder.mask_tokens.weight", [n_enc_out_chans, n_pt_embd]);

			}
			return model;
		}

		static ggml_tensor* GetTensorFromName(ggml_context* context, List<PickleLoader.CommonTensor> tensors, string name, ggml_type type = ggml_type.GGML_TYPE_F32)
		{
			var tensor = tensors.Find(x => x.Name == name);
			if (tensor == null)
			{
				throw new Exception($"tensor {name} not found");
			}
			byte[] bytes = PickleLoader.ReadByteFromFile(tensor);
			bytes = TransData(bytes, tensor.Type, type);
			long[] ne = new long[tensor.Shape.Count];
			for (int i = 0; i < tensor.Shape.Count; i++)
			{
				ne[i] = (long)tensor.Shape[i];
			}
			ggml_tensor* t = Native.ggml_new_tensor(context, type, ne.Length, ne);
			t->data = Marshal.AllocHGlobal(bytes.Length);
			Marshal.Copy(bytes, 0, t->data, bytes.Length);
			Native.ggml_set_name(t, tensor.Name);
			return t;
		}

		static ggml_tensor* GetTensorFromName(ggml_context* context, List<PickleLoader.CommonTensor> tensors, string name, long[] shape, ggml_type type = ggml_type.GGML_TYPE_F32)
		{
			if (shape == null)
			{
				throw new ArgumentException("Shape is empty");
			}
			if (shape.Length < 0 || shape.Length > 4)
			{
				throw new ArgumentException("Shape is out of range");
			}
			var tensor = tensors.Find(x => x.Name == name);
			if (tensor == null)
			{
				throw new Exception($"tensor {name} not found");
			}
			ulong orgShapeMut = 1;
			foreach (ulong a in tensor.Shape)
			{
				orgShapeMut *= a;
			};
			ulong desShapeMut = 1;
			foreach (ulong a in shape)
			{
				desShapeMut *= a;
			}
			if (orgShapeMut != desShapeMut)
			{
				throw new Exception("Shape not same");
			}
			byte[] bytes = PickleLoader.ReadByteFromFile(tensor);
			bytes = TransData(bytes, tensor.Type, type);
			ggml_tensor* t = Native.ggml_new_tensor(context, type, shape.Length, shape);
			Marshal.Copy(bytes, 0, t->data, bytes.Length);
			Native.ggml_set_name(t, tensor.Name);
			return t;
		}

		static byte[] TransData(byte[] data, ggml_type orgType, ggml_type desType)
		{
			if (orgType == desType)
			{
				return data;
			}
			if (orgType != ggml_type.GGML_TYPE_F16 && orgType != ggml_type.GGML_TYPE_BF16 && orgType != ggml_type.GGML_TYPE_F32)
			{
				throw new ArgumentException("Org Type not support");
			}
			if (desType != ggml_type.GGML_TYPE_F16 && desType != ggml_type.GGML_TYPE_BF16 && desType != ggml_type.GGML_TYPE_F32)
			{
				throw new ArgumentException("Des Type not support");
			}

			if (orgType == ggml_type.GGML_TYPE_BF16)
			{
				if (desType == ggml_type.GGML_TYPE_F16)
				{
					for (int j = 0; j < data.Length; j += 2)
					{
						ushort data16 = (ushort)(data[j] | (data[j + 1] << 8));
						float data32 = Native.ggml_bf16_to_fp32(data16);
						data16 = Native.ggml_fp32_to_fp16(data32);
						byte[] bytes = BitConverter.GetBytes(data16);
						data[j] = bytes[0];
						data[j + 1] = bytes[1];
					}
					return data;
				}
				else if (desType == ggml_type.GGML_TYPE_F32)
				{
					byte[] f32bytes = new byte[data.Length * 2];
					for (int j = 0; j < data.Length / 2; j++)
					{
						ushort data16 = (ushort)(data[j * 2] | (data[j * 2 + 1] << 8));
						float data32 = Native.ggml_bf16_to_fp32(data16);
						byte[] bytes = BitConverter.GetBytes(data32);
						f32bytes[j * 4] = bytes[0];
						f32bytes[j * 4 + 1] = bytes[1];
						f32bytes[j * 4 + 2] = bytes[2];
						f32bytes[j * 4 + 3] = bytes[3];
					}
					return f32bytes;
				}
			}
			else if (orgType == ggml_type.GGML_TYPE_F32)
			{
				byte[] bytes = new byte[data.Length / 2];
				for (int j = 0; j < data.Length / 4; j++)
				{
					float f32 = BitConverter.ToSingle(data, j * 4);
					ushort f16Data = 0;
					if (desType == ggml_type.GGML_TYPE_BF16)
					{
						f16Data = Native.ggml_fp32_to_bf16(f32);
					}
					else if (desType == ggml_type.GGML_TYPE_F16)
					{
						f16Data = Native.ggml_fp32_to_fp16(f32);
					}
					byte[] bt = BitConverter.GetBytes(f16Data);
					bytes[j * 2] = bt[0];
					bytes[j * 2 + 1] = bt[1];
				}
				return bytes;
			}
			else if (orgType == ggml_type.GGML_TYPE_F16)
			{
				if (desType == ggml_type.GGML_TYPE_BF16)
				{
					for (int j = 0; j < data.Length; j += 2)
					{
						ushort data16 = (ushort)(data[j] | (data[j + 1] << 8));
						float data32 = Native.ggml_fp16_to_fp32(data16);
						data16 = Native.ggml_fp32_to_bf16(data32);
						byte[] bytes = BitConverter.GetBytes(data16);
						data[j] = bytes[0];
						data[j + 1] = bytes[1];
					}
					return data;
				}
				else if (desType == ggml_type.GGML_TYPE_F32)
				{
					byte[] f32bytes = new byte[data.Length * 2];
					for (int j = 0; j < data.Length / 2; j++)
					{
						ushort data16 = (ushort)(data[j * 2] | (data[j * 2 + 1] << 8));
						float data32 = Native.ggml_fp16_to_fp32(data16);
						byte[] bytes = BitConverter.GetBytes(data32);
						f32bytes[j * 4] = bytes[0];
						f32bytes[j * 4 + 1] = bytes[1];
						f32bytes[j * 4 + 2] = bytes[2];
						f32bytes[j * 4 + 3] = bytes[3];
					}
					return f32bytes;
				}
			}
			throw new ArgumentException("Not support!");
		}

		struct prompt_encoder_result
		{
			public ggml_tensor* embd_prompt_sparse;
			public ggml_tensor* embd_prompt_dense;
		};

		static ggml_tensor* sam_fill_dense_pe(sam_model model, ggml_context* ctx0, ggml_cgraph* gf, sam_state state)
		{
			sam_hparams hparams = model.hparams;
			sam_encoder_prompt enc = model.enc_prompt;


			int n_img_embd = hparams.n_img_embd();
			ggml_tensor* xy_embed_stacked = Native.ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_F32, 2, n_img_embd, n_img_embd);
			Native.ggml_set_name(xy_embed_stacked, "xy_embed_stacked");
			Native.ggml_set_input(xy_embed_stacked);

			ggml_tensor* cur = Native.ggml_mul_mat(ctx0, Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, enc.pe)), xy_embed_stacked);

			cur = Native.ggml_scale(ctx0, cur, 2.0f * MathF.PI);

			// concat
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
			{
				ggml_tensor* t_sin = Native.ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, null);
				ggml_tensor* t_cos = Native.ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, null);

				cur = Native.ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1], cur->ne[2]);

				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, t_sin, Native.ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], 0)));
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, t_cos, Native.ggml_view_3d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], t_sin->ne[2], cur->nb[1], cur->nb[2], t_sin->nb[1])));
			}

			ggml_tensor* pe_img_dense = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, cur, 2, 0, 1, 3));
			Native.ggml_build_forward_expand(gf, pe_img_dense);

			return pe_img_dense;
		}

		static prompt_encoder_result sam_encode_prompt(sam_model model, ggml_context* ctx0, ggml_cgraph* gf, sam_state state)
		{
			sam_hparams hparams = model.hparams;
			sam_encoder_prompt enc = model.enc_prompt;


			ggml_tensor* inp = Native.ggml_new_tensor_2d(ctx0, ggml_type.GGML_TYPE_F32, 2, 2);
			Native.ggml_set_name(inp, "prompt_input");
			Native.ggml_set_input(inp);


			ggml_tensor* cur = Native.ggml_mul_mat(ctx0, Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, enc.pe)), inp);

			cur = Native.ggml_scale(ctx0, cur, 2.0f * MathF.PI);

			// concat
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
			{
				ggml_tensor* t_sin = Native.ggml_map_custom1(ctx0, cur, ggml_sam_sin, GGML_N_TASKS_MAX, null);
				ggml_tensor* t_cos = Native.ggml_map_custom1(ctx0, cur, ggml_sam_cos, GGML_N_TASKS_MAX, null);

				cur = Native.ggml_new_tensor_2d(ctx0, ggml_type.GGML_TYPE_F32, t_sin->ne[0] + t_cos->ne[0], cur->ne[1]);

				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, t_sin, Native.ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], 0)));
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, t_cos, Native.ggml_view_2d(ctx0, cur, t_sin->ne[0], t_sin->ne[1], cur->nb[1], t_sin->nb[1])));

				// overwrite label == -1 with not_a_point_embed.weight
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
				// TODO: extend for multiple points
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, enc.not_a_pt_embd_w, Native.ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], cur->nb[1])));
			}

			// add point_embeddings[1] to label == 1
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
			ggml_tensor* v = Native.ggml_view_2d(ctx0, cur, cur->ne[0], 1, cur->nb[1], 0);
			Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, Native.ggml_add_inplace(ctx0, v, enc.pt_embd[1]), v));

			ggml_tensor* embd_prompt_sparse = cur;
			Native.ggml_build_forward_expand(gf, embd_prompt_sparse);

			ggml_tensor* embd_prompt_dense = Native.ggml_repeat(ctx0,
				Native.ggml_cont(ctx0,
					Native.ggml_view_3d(ctx0, enc.no_mask_embd_w,
						  1, 1, enc.no_mask_embd_w->ne[0], enc.no_mask_embd_w->nb[0], enc.no_mask_embd_w->nb[0], 0)),
				Native.ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_F32, hparams.n_img_embd(), hparams.n_img_embd(), hparams.n_enc_out_chans));

			Native.ggml_build_forward_expand(gf, embd_prompt_dense);

			//printf("used_mem = %zu\n", ggml_used_mem(ctx0));

			prompt_encoder_result res;
			res.embd_prompt_sparse = embd_prompt_sparse;
			res.embd_prompt_dense = embd_prompt_dense;
			return res;
		}

		static ggml_tensor* sam_decode_mask_transformer_attn(sam_layer_dec_transformer_attn attn, ggml_tensor* queries, ggml_tensor* keys, ggml_tensor* values, ggml_context* ctx0, sam_model model)
		{
			sam_hparams hparams = model.hparams;
			int n_head = hparams.n_dec_heads;

			ggml_tensor* Qcur;
			ggml_tensor* Kcur;
			ggml_tensor* Vcur;

			Qcur = Native.ggml_mul_mat(ctx0, attn.q_w, queries);
			Qcur = Native.ggml_add_inplace(ctx0, Qcur, attn.q_b);

			Kcur = Native.ggml_mul_mat(ctx0, attn.k_w, keys);
			Kcur = Native.ggml_add_inplace(ctx0, Kcur, attn.k_b);

			Vcur = Native.ggml_mul_mat(ctx0, attn.v_w, values);
			Vcur = Native.ggml_add_inplace(ctx0, Vcur, attn.v_b);

			ggml_tensor* Q;
			ggml_tensor* K;
			ggml_tensor* V;

			Q = Native.ggml_reshape_4d(ctx0, Qcur, Qcur->ne[0] / n_head, n_head, Qcur->ne[1], Qcur->ne[2]);
			Q = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, Q, 0, 2, 1, 3));

			K = Native.ggml_reshape_4d(ctx0, Kcur, Kcur->ne[0] / n_head, n_head, Kcur->ne[1], Kcur->ne[2]);
			K = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, K, 0, 2, 1, 3));

			V = Native.ggml_reshape_4d(ctx0, Vcur, Vcur->ne[0] / n_head, n_head, Vcur->ne[1], Vcur->ne[2]);
			V = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, V, 0, 2, 1, 3));

			// Q * K
			ggml_tensor* KQ = Native.ggml_mul_mat(ctx0, K, Q);

			ggml_tensor* KQ_scaled = Native.ggml_scale_inplace(ctx0, KQ, 1.0f / MathF.Sqrt(Q->ne[0]));

			ggml_tensor* KQ_soft_max = Native.ggml_soft_max_inplace(ctx0, KQ_scaled);

			ggml_tensor* KQV = Native.ggml_mul_mat(ctx0, KQ_soft_max, Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, V)));

			ggml_tensor* KQV_merged = Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, KQV));
			KQV_merged = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, KQV_merged, 0, 2, 1, 3));
			KQV_merged = Native.ggml_reshape_3d(ctx0, KQV_merged, KQV_merged->ne[0] * KQV_merged->ne[1], KQV_merged->ne[2], KQV_merged->ne[3]);
			KQV_merged = Native.ggml_mul_mat(ctx0, attn.out_w, KQV_merged);
			KQV_merged = Native.ggml_add_inplace(ctx0, KQV_merged, attn.out_b);

			return KQV_merged;
		}

		static ggml_tensor* sam_decode_mask_mlp_relu_3(ggml_tensor* input, ggml_tensor* w_0, ggml_tensor* b_0, ggml_tensor* w_1, ggml_tensor* b_1, ggml_tensor* w_2, ggml_tensor* b_2, ggml_context* ctx0)
		{
			ggml_tensor* cur;
			cur = Native.ggml_mul_mat(ctx0, w_0, input);
			cur = Native.ggml_add_inplace(ctx0, cur, b_0);

			cur = Native.ggml_relu_inplace(ctx0, cur);

			cur = Native.ggml_mul_mat(ctx0, w_1, cur);
			cur = Native.ggml_add_inplace(ctx0, cur, b_1);

			cur = Native.ggml_relu_inplace(ctx0, cur);

			cur = Native.ggml_mul_mat(ctx0, w_2, cur);
			cur = Native.ggml_add_inplace(ctx0, cur, b_2);

			return cur;
		}

		static ggml_tensor* sam_layer_norm_2d(ggml_context* ctx0, ggml_tensor* layer, int n_channels, ggml_tensor* w, ggml_tensor* b, float eps)
		{
			// LayerNorm2d
			// normalize along channel dimmension
			// TODO: better implementation
			layer = Native.ggml_permute(ctx0,
					Native.ggml_norm(ctx0, Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, layer, 1, 2, 0, 3)), eps),
						2, 0, 1, 3);

			layer = Native.ggml_add(ctx0,
					 Native.ggml_mul(ctx0,
						 Native.ggml_repeat(ctx0, Native.ggml_reshape_3d(ctx0, w, 1, 1, n_channels), layer),
						  layer),
					Native.ggml_repeat(ctx0, Native.ggml_reshape_3d(ctx0, b, 1, 1, n_channels), layer));

			return layer;
		}

		static bool sam_decode_mask(sam_model model, prompt_encoder_result prompt, ggml_tensor* pe_img, ggml_context* ctx0, ggml_cgraph* gf, sam_state state)
		{
			sam_hparams hparams = model.hparams;
			sam_decoder_mask dec = model.dec;
			int n_img_embd = hparams.n_img_embd();

			ggml_tensor* tokens;
			{
				// Concatenate output tokens
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
				ggml_tensor* sparse = prompt.embd_prompt_sparse;

				tokens = Native.ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_F32, dec.iou_token_w->ne[0], dec.iou_token_w->ne[1] + dec.mask_tokens_w->ne[1] + sparse->ne[1], sparse->ne[2]);

				ulong[] offsets = { 0, (ulong)dec.iou_token_w->ne[1] * tokens->nb[1], (ulong)dec.iou_token_w->ne[1] * tokens->nb[1] + (ulong)dec.mask_tokens_w->ne[1] * tokens->nb[1] };
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, dec.iou_token_w, Native.ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.iou_token_w->ne[1], tokens->nb[1], offsets[0])));
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, dec.mask_tokens_w, Native.ggml_view_2d(ctx0, tokens, tokens->ne[0], dec.mask_tokens_w->ne[1], tokens->nb[1], offsets[1])));
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, sparse, Native.ggml_view_2d(ctx0, tokens, tokens->ne[0], sparse->ne[1], tokens->nb[1], offsets[2])));
				// TODO: Sparse prompt embeddings can have more than one point
			}


			ggml_tensor* src;
			ggml_tensor* pos_src;
			long[] srcNE = { 0, 0, 0, 0 };
			{
				// Expand per-image data in the batch direction to be per-mask
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
				src = Native.ggml_new_tensor_4d(ctx0, ggml_type.GGML_TYPE_F32, state.embd_img->ne[0], state.embd_img->ne[1], state.embd_img->ne[2], tokens->ne[2]);

				src = Native.ggml_add(ctx0,
					Native.ggml_repeat(ctx0,
						state.embd_img,
						src),
					prompt.embd_prompt_dense);

				srcNE[0] = src->ne[0];
				srcNE[1] = src->ne[1];
				srcNE[2] = src->ne[2];
				srcNE[3] = src->ne[3];

				// flatten & permute
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
				src = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0,
					Native.ggml_view_3d(ctx0,
						src,
						src->ne[0] * src->ne[1],
						src->ne[2],
						src->ne[3],
						src->nb[2],
						src->nb[3],
						0),
					1, 0, 2, 3));

				pos_src = Native.ggml_new_tensor_4d(ctx0, ggml_type.GGML_TYPE_F32, pe_img->ne[0], pe_img->ne[1], pe_img->ne[2], tokens->ne[2]);
				pos_src = Native.ggml_repeat(ctx0,
					pe_img,
					pos_src);

				// flatten & permute
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
				pos_src = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0,
					Native.ggml_view_3d(ctx0,
						pos_src,
						pos_src->ne[0] * pos_src->ne[1],
						pos_src->ne[2],
						pos_src->ne[3],
						pos_src->nb[2],
						pos_src->nb[3],
							0),
						1, 0, 2, 3));
			}

			ggml_tensor* queries = tokens;
			ggml_tensor* keys = src;
			{
				// Run the transformer
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
				for (int i = 0; i < model.dec.transformer_layers.Length; ++i)
				{
					sam_layer_dec_transformer tfm_layer = model.dec.transformer_layers[i];

					// Self attention block
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
					bool skip_first_layer_pe = i == 0;
					if (skip_first_layer_pe)
					{
						queries = sam_decode_mask_transformer_attn(tfm_layer.self_attn, queries, queries, queries, ctx0, model);
					}
					else
					{

						ggml_tensor* q_0 = Native.ggml_add(ctx0, queries, tokens);

						ggml_tensor* self_attn = sam_decode_mask_transformer_attn(tfm_layer.self_attn, q_0, q_0, queries, ctx0, model);
						queries = Native.ggml_add(ctx0, queries, self_attn);
					}

					queries = Native.ggml_norm(ctx0, queries, hparams.eps_decoder_transformer);
					queries = Native.ggml_add_inplace(ctx0,
						Native.ggml_mul(ctx0, queries, tfm_layer.norm1_w),
							tfm_layer.norm1_b);

					// Cross attention block, tokens attending to image embedding
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
					ggml_tensor* q_1 = Native.ggml_add(ctx0, queries, tokens);
					ggml_tensor* k_1 = Native.ggml_add(ctx0, keys, pos_src);

					ggml_tensor* cross_attn_token_to_img = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_token_to_img, q_1, k_1, keys, ctx0, model);

					queries = Native.ggml_add_inplace(ctx0, queries, cross_attn_token_to_img);
					queries = Native.ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
					queries = Native.ggml_add_inplace(ctx0,
							Native.ggml_mul(ctx0, queries, tfm_layer.norm2_w),
							tfm_layer.norm2_b);

					// MLP block
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
					ggml_tensor* mlp_out = Native.ggml_mul_mat(ctx0,
					  tfm_layer.mlp_lin1_w,
					  queries);

					mlp_out = Native.ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin1_b);

					// RELU activation
					mlp_out = Native.ggml_relu_inplace(ctx0, mlp_out);
					mlp_out = Native.ggml_mul_mat(ctx0, tfm_layer.mlp_lin2_w, mlp_out);

					mlp_out = Native.ggml_add_inplace(ctx0, mlp_out, tfm_layer.mlp_lin2_b);

					queries = Native.ggml_add_inplace(ctx0, queries, mlp_out);
					queries = Native.ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
					queries = Native.ggml_add_inplace(ctx0,
							Native.ggml_mul(ctx0, queries, tfm_layer.norm3_w),
							tfm_layer.norm3_b);

					// Cross attention block, image embedding attending to tokens
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
					ggml_tensor* q_2 = Native.ggml_add(ctx0, queries, tokens);
					ggml_tensor* k_2 = Native.ggml_add(ctx0, keys, pos_src);

					ggml_tensor* cross_attn_img_to_token = sam_decode_mask_transformer_attn(tfm_layer.cross_attn_img_to_token, k_2, q_2, queries, ctx0, model);
					keys = Native.ggml_add_inplace(ctx0, keys, cross_attn_img_to_token);
					keys = Native.ggml_norm_inplace(ctx0, keys, hparams.eps_decoder_transformer);
					keys = Native.ggml_add_inplace(ctx0,
							Native.ggml_mul(ctx0, keys, tfm_layer.norm4_w),
							tfm_layer.norm4_b);
				}

				// Apply the final attention layer from the points to the image
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
				ggml_tensor* q = Native.ggml_add(ctx0, queries, tokens);
				ggml_tensor* k = Native.ggml_add(ctx0, keys, pos_src);

				ggml_tensor* final_attn_token_to_img = sam_decode_mask_transformer_attn(dec.transformer_final_attn_token_to_img, q, k, keys, ctx0, model);

				queries = Native.ggml_add_inplace(ctx0, queries, final_attn_token_to_img);
				queries = Native.ggml_norm_inplace(ctx0, queries, hparams.eps_decoder_transformer);
				queries = Native.ggml_add_inplace(ctx0,
						Native.ggml_mul(ctx0, queries, dec.transformer_norm_final_w),
						dec.transformer_norm_final_b);
			}


			ggml_tensor* iou_pred = Native.ggml_view_2d(ctx0, queries, queries->ne[0], queries->ne[2], queries->nb[2], 0);
			const int num_mask_tokens = 4; // num_multimask_outputs + 1
			ggml_tensor* mask_tokens_out = Native.ggml_view_3d(ctx0, queries, queries->ne[0], num_mask_tokens, queries->ne[2], queries->nb[1], num_mask_tokens * queries->nb[1], queries->nb[1]);

			// Upscale mask embeddings and predict masks using the mask tokens
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
			keys = Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, keys));
			keys = Native.ggml_view_4d(ctx0, keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], (ulong)srcNE[0] * keys->nb[0], keys->nb[1], keys->nb[2], 0);
			// ggml_build_forward_expand(gf, keys);
			ggml_tensor* upscaled_embedding;
			{
				// ConvTranspose2d
				keys = Native.ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_0_w, keys, 2);
				keys = Native.ggml_add_inplace(ctx0, keys, Native.ggml_repeat(ctx0,
											Native.ggml_reshape_3d(ctx0, dec.output_upscaling_0_b, 1, 1, dec.output_upscaling_0_b->ne[0]),
											 keys));

				keys = sam_layer_norm_2d(ctx0, keys, n_img_embd, dec.output_upscaling_1_w, dec.output_upscaling_1_b, hparams.eps);

				// GELU activation
				keys = Native.ggml_gelu_inplace(ctx0, keys);

				// ConvTranspose2d
				keys = Native.ggml_conv_transpose_2d_p0(ctx0, dec.output_upscaling_3_w, keys, 2);
				keys = Native.ggml_add_inplace(ctx0, Native.ggml_repeat(ctx0,
										Native.ggml_reshape_3d(ctx0, dec.output_upscaling_3_b, 1, 1, dec.output_upscaling_3_b->ne[0]),
										keys), keys);
				// GELU activation
				keys = Native.ggml_gelu_inplace(ctx0, keys);
				upscaled_embedding = Native.ggml_reshape_3d(ctx0, keys, keys->ne[0] * keys->ne[1], keys->ne[2], keys->ne[3]);
				upscaled_embedding = Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, upscaled_embedding)); // TODO: Shouldn't be needed
			}

			ggml_tensor* hyper_in = Native.ggml_new_tensor_3d(ctx0, ggml_type.GGML_TYPE_F32, n_img_embd / 2, num_mask_tokens, mask_tokens_out->ne[2]);

			for (int i = 0; i < num_mask_tokens; ++i)
			{
				sam_layer_dec_output_hypernet_mlps mlp = dec.output_hypernet_mlps[i];

				ggml_tensor* input = Native.ggml_view_2d(ctx0, mask_tokens_out, mask_tokens_out->ne[0], mask_tokens_out->ne[2], mask_tokens_out->nb[1], (ulong)i * mask_tokens_out->nb[1]);
				ggml_tensor* output = sam_decode_mask_mlp_relu_3(input, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2, ctx0);
				Native.ggml_build_forward_expand(gf, Native.ggml_cpy(ctx0, output, Native.ggml_view_2d(ctx0, hyper_in, hyper_in->ne[0], hyper_in->ne[2], hyper_in->nb[1], (ulong)i * hyper_in->nb[1])));
			}

			ggml_tensor* masks = Native.ggml_mul_mat(ctx0, hyper_in, upscaled_embedding);
			masks = Native.ggml_cont(ctx0, Native.ggml_transpose(ctx0, masks)); // TODO: Shouldn't be needed
			masks = Native.ggml_reshape_4d(ctx0, masks, keys->ne[0], keys->ne[1], masks->ne[1], keys->ne[3]);

			// Generate mask quality predictions
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
			iou_pred = sam_decode_mask_mlp_relu_3(iou_pred, dec.iou_prediction_head_0_w, dec.iou_prediction_head_0_b, dec.iou_prediction_head_1_w, dec.iou_prediction_head_1_b, dec.iou_prediction_head_2_w, dec.iou_prediction_head_2_b, ctx0);

			// Select the correct mask or masks for output
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
			iou_pred = Native.ggml_cpy(state.ctx, Native.ggml_view_1d(ctx0, iou_pred, iou_pred->ne[0] - 1, iou_pred->nb[0]), state.iou_predictions);
			masks = Native.ggml_view_4d(ctx0, masks, masks->ne[0], masks->ne[1], masks->ne[2] - 1, masks->ne[3], masks->nb[1], masks->nb[2], masks->nb[3], masks->nb[2] /* offset*/);
			masks = Native.ggml_cpy(state.ctx, masks, state.low_res_masks);
			Native.ggml_build_forward_expand(gf, masks);
			Native.ggml_build_forward_expand(gf, iou_pred);
			ggml_disconnect_node_from_graph(state.low_res_masks);
			ggml_disconnect_node_from_graph(state.iou_predictions);

			return true;
		}

		static ggml_cgraph* sam_build_fast_graph(sam_model model, sam_state state, int nx, int ny, sam_point point)
		{
			ulong size = Native.ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + Native.ggml_graph_overhead();
			ggml_init_params ggml_params = new ggml_init_params()
			{
				mem_size = 1024 * 1024 * 1024u,
				mem_buffer = IntPtr.Zero,
				no_alloc = true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
			};

			ggml_context* ctx0 = Native.ggml_init(ggml_params);
			ggml_cgraph* gf = Native.ggml_new_graph(ctx0);

			prompt_encoder_result enc_res = sam_encode_prompt(model, ctx0, gf, state);
			if (enc_res.embd_prompt_sparse == null || enc_res.embd_prompt_dense == null)
			{
				throw new Exception($"failed to encode prompt ({point.x}, {point.y})");
			}

			ggml_tensor* pe_img_dense = sam_fill_dense_pe(model, ctx0, gf, state);
			if (pe_img_dense == null)
			{
				throw new Exception("failed to get dense positional encoding");
			}

			if (!sam_decode_mask(model, enc_res, pe_img_dense, ctx0, gf, state))
			{
				throw new Exception("failed to decode mask");
			}

			//Native.ggml_free(ctx0);

			Native.ggml_gallocr_alloc_graph(state.allocr, gf);

			// from sam_encode_prompt
			{
				// transform points
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
				{
					int nmax = Math.Max(nx, ny);

					float scale = model.hparams.n_img_size() / (float)nmax;

					int nx_new = (int)(nx * scale + 0.5f);
					int ny_new = (int)(ny * scale + 0.5f);

					point.x = point.x * ((float)(nx_new) / nx) + 0.5f;
					point.y = point.y * ((float)(ny_new) / ny) + 0.5f;
				}


				ggml_tensor* inp = Native.ggml_graph_get_tensor(gf, "prompt_input");
				// set the input by converting the [0, 1] coordinates to [-1, 1]
				float* data = (float*)inp->data;

				data[0] = 2.0f * (point.x / model.hparams.n_img_size()) - 1.0f;
				data[1] = 2.0f * (point.y / model.hparams.n_img_size()) - 1.0f;

				// padding
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
				data[2] = 2.0f * (0.0f) - 1.0f;
				data[3] = 2.0f * (0.0f) - 1.0f;
			}

			// from sam_fill_dense_pe
			{

				ggml_tensor* xy_embed_stacked = Native.ggml_graph_get_tensor(gf, "xy_embed_stacked");
				int n_img_embd = model.hparams.n_img_embd();
				float n_img_embd_inv = 1.0f / n_img_embd;
				float* data = (float*)Native.ggml_get_data(xy_embed_stacked);
				for (int i = 0; i < n_img_embd; ++i)
				{
					int row = 2 * i * n_img_embd;
					float y_val = 2 * (i + 0.5f) * n_img_embd_inv - 1;
					for (int j = 0; j < n_img_embd; ++j)
					{
						float x_val = 2 * (j + 0.5f) * n_img_embd_inv - 1;
						data[row + 2 * j + 0] = x_val;
						data[row + 2 * j + 1] = y_val;
					}
				}
			}
			return gf;
		}

		static ggml_cgraph* sam_encode_image(sam_model model, sam_state state, sam_image_f32 img)
		{
			sam_hparams hparams = model.hparams;
			sam_encoder_image enc = model.enc_img;

			int n_enc_state = hparams.n_enc_state;
			int n_enc_layer = hparams.n_enc_layer;
			int n_enc_head = hparams.n_enc_head;
			int n_enc_head_dim = hparams.n_enc_head_dim();
			int n_enc_out_chans = hparams.n_enc_out_chans;
			int n_img_size = hparams.n_img_size();
			int n_window_size = hparams.n_window_size();


			ggml_init_params ggml_params = new ggml_init_params()
			{
				mem_size = (ulong)state.buf_compute_img_enc.Length,
				mem_buffer = Marshal.UnsafeAddrOfPinnedArrayElement(state.buf_compute_img_enc, 0),
				no_alloc = true, // skip allocating as we use ggml_alloc to allocate exact memory requirements
			};

			ggml_context* ctx0 = Native.ggml_init(ggml_params);
			ggml_cgraph* gf = Native.ggml_new_graph(ctx0);

			ggml_tensor* inp = Native.ggml_new_tensor_4d(ctx0, ggml_type.GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
			Native.ggml_set_name(inp, "inp");
			Native.ggml_set_input(inp);

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
			ggml_tensor* cur = Native.ggml_conv_2d_sk_p0(ctx0, enc.proj_w, inp);
			cur = Native.ggml_add_inplace(ctx0,
					cur,
					Native.ggml_repeat(ctx0, enc.proj_b, cur));

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
			// keep in F32
			cur = Native.ggml_cont(ctx0,
					Native.ggml_permute(ctx0, cur, 1, 2, 0, 3));

			// convert to F16
			//cur = ggml_cpy(ctx0,
			//        ggml_permute(ctx0, cur, 1, 2, 0, 3),
			//        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, n_enc_state, n_img_embd, n_img_embd));

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
			cur = Native.ggml_add_inplace(ctx0, cur, enc.pe);

			ggml_tensor* inpL = cur;

			for (int il = 0; il < n_enc_layer; ++il)
			{
				sam_layer_enc layer = enc.layers[il];

				// norm
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
				{
					cur = Native.ggml_norm(ctx0, inpL, hparams.eps);

					// cur = ln_0_w*cur + ln_0_b
					cur = Native.ggml_mul(ctx0, cur, layer.norm1_w);
					cur = Native.ggml_add_inplace(ctx0, cur, layer.norm1_b);
				}

				long w0 = cur->ne[1];
				long h0 = cur->ne[2];

				if (hparams.is_global_attn(il) == false)
				{
					// local attention layer - apply window partition
					// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
					cur = Native.ggml_win_part(ctx0, cur, n_window_size);
				}
				long W = cur->ne[1];
				long H = cur->ne[2];

				// self-attention
				{
					cur = Native.ggml_mul_mat(ctx0, layer.qkv_w, cur);
					cur = Native.ggml_add_inplace(ctx0, cur, layer.qkv_b);

					// split qkv into separate tensors
					// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
					long B = cur->ne[3];

					cur = Native.ggml_reshape_4d(ctx0, cur, n_enc_state, 3, W * H, B);
					cur = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, cur, 0, 3, 1, 2));


					ggml_tensor* Q;
					ggml_tensor* K;
					ggml_tensor* V;

					Q = Native.ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 0 * cur->nb[3]);
					Q = Native.ggml_reshape_4d(ctx0, Q, n_enc_head_dim, n_enc_head, W * H, B);
					Q = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, Q, 0, 2, 1, 3));
					Q = Native.ggml_reshape_3d(ctx0, Q, n_enc_head_dim, W * H, B * n_enc_head);

					K = Native.ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 1 * cur->nb[3]);
					K = Native.ggml_reshape_4d(ctx0, K, n_enc_head_dim, n_enc_head, W * H, B);
					K = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, K, 0, 2, 1, 3));
					K = Native.ggml_reshape_3d(ctx0, K, n_enc_head_dim, W * H, B * n_enc_head);
					V = Native.ggml_view_3d(ctx0, cur, n_enc_state, W * H, B, cur->nb[1], cur->nb[2], 2 * cur->nb[3]);
					V = Native.ggml_reshape_4d(ctx0, V, n_enc_head_dim, n_enc_head, W * H, B);
					V = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, V, 1, 2, 0, 3)); // transposed
					V = Native.ggml_reshape_3d(ctx0, V, W * H, n_enc_head_dim, B * n_enc_head);

					ggml_tensor* KQ = Native.ggml_mul_mat(ctx0, K, Q);

					ggml_tensor* KQ_scaled = Native.ggml_scale_inplace(ctx0, KQ, 1.0f / MathF.Sqrt(n_enc_head_dim));

					ggml_tensor* rw = Native.ggml_get_rel_pos(ctx0, layer.rel_pos_w, (int)W, (int)W);
					ggml_tensor* rh = Native.ggml_get_rel_pos(ctx0, layer.rel_pos_h, (int)H, (int)H);

					ggml_tensor* q_r = Native.ggml_reshape_4d(ctx0, Q, n_enc_head_dim, W, H, B * n_enc_head);

					ggml_tensor* rel_w = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0,
							  Native.ggml_mul_mat(ctx0,
								  rw,
								 Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, q_r, 0, 2, 1, 3))),
							  0, 2, 1, 3));
					ggml_tensor* rel_h = Native.ggml_mul_mat(ctx0, rh, q_r);

					ggml_tensor* attn = Native.ggml_add_rel_pos_inplace(ctx0, KQ_scaled, rel_w, rel_h);

					ggml_tensor* KQ_soft_max = Native.ggml_soft_max_inplace(ctx0, attn);

					ggml_tensor* KQV = Native.ggml_mul_mat(ctx0, V, KQ_soft_max);

					cur =
						Native.ggml_reshape_4d(ctx0,
								Native.ggml_cont(ctx0,
									Native.ggml_permute(ctx0,
										Native.ggml_reshape_4d(ctx0, KQV, n_enc_head_dim, W * H, n_enc_head, B),
					0, 2, 1, 3)),
					n_enc_state, W, H, B);

					cur = Native.ggml_mul_mat(ctx0, layer.proj_w, cur);
					cur = Native.ggml_add_inplace(ctx0, cur, layer.proj_b);
				}

				if (hparams.is_global_attn(il) == false)
				{
					// local attention layer - reverse window partition
					cur = Native.ggml_win_unpart(ctx0, cur, (int)w0, (int)h0, n_window_size);
				}

				cur = Native.ggml_add_inplace(ctx0, cur, inpL);

				ggml_tensor* inpFF = cur;

				// feed-forward network
				{
					// norm
					{
						cur = Native.ggml_norm(ctx0, inpFF, hparams.eps);

						// cur = mlp_ln_w*cur + mlp_ln_b
						cur = Native.ggml_mul(ctx0, cur, layer.norm2_w);
						cur = Native.ggml_add_inplace(ctx0, cur, layer.norm2_b);
					}

					// fully connected
					cur = Native.ggml_mul_mat(ctx0, layer.mlp_lin1_w, cur);
					cur = Native.ggml_add_inplace(ctx0, cur, layer.mlp_lin1_b);

					// GELU activation
					cur = Native.ggml_gelu(ctx0, cur);

					// projection
					cur = Native.ggml_mul_mat(ctx0, layer.mlp_lin2_w, cur);
					cur = Native.ggml_add_inplace(ctx0, cur, layer.mlp_lin2_b);
				}

				inpL = Native.ggml_add(ctx0, cur, inpFF);
			}

			cur = Native.ggml_cont(ctx0, Native.ggml_permute(ctx0, inpL, 2, 0, 1, 3));
			cur = Native.ggml_conv_2d_sk_p0(ctx0, enc.neck_conv_0, cur);
			cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_0_w, enc.neck_norm_0_b, hparams.eps);
			cur = Native.ggml_conv_2d_s1_ph(ctx0, enc.neck_conv_1, cur);
			cur = sam_layer_norm_2d(ctx0, cur, n_enc_out_chans, enc.neck_norm_1_w, enc.neck_norm_1_b, hparams.eps);

			cur = Native.ggml_cpy(ctx0, cur, state.embd_img);

			Native.ggml_build_forward_expand(gf, cur);
			ggml_disconnect_node_from_graph(state.embd_img);

			//ggml_graph_print(&gf);

			Native.ggml_free(ctx0);

			Native.ggml_gallocr_alloc_graph(state.allocr, gf);
			{
				ggml_tensor* inpt = Native.ggml_graph_get_tensor(gf, "inp");
				float* data = (float*)Native.ggml_get_data(inpt);

				int nx = img.nx;
				int ny = img.ny;
				int n = nx * ny;

				if (!(nx == n_img_size && ny == n_img_size))
				{
					throw new Exception("nx == n_img_size && ny == n_img_size wrong");
				}

				for (int k = 0; k < 3; k++)
				{
					for (int y = 0; y < ny; y++)
					{
						for (int x = 0; x < nx; x++)
						{
							data[k * n + y * nx + x] = img.data[3 * (y * nx + x) + k];
						}
					}
				}
			}

			return gf;
		}

		static bool sam_write_masks(sam_hparams hparams, int nx, int ny, sam_state state, string fname)
		{
			if (state.low_res_masks->ne[2] == 0) return true;
			if (state.low_res_masks->ne[2] != state.iou_predictions->ne[0])
			{
				throw new ArgumentException($"Error: number of masks ({state.low_res_masks->ne[2]}) does not match number of iou predictions ({state.iou_predictions->ne[0]})");
			}

			int n_img_size = hparams.n_img_size();
			float mask_threshold = hparams.mask_threshold;
			float iou_threshold = hparams.iou_threshold;
			float stability_score_threshold = hparams.stability_score_threshold;
			float intersection_threshold = mask_threshold + hparams.stability_score_offset;
			float union_threshold = mask_threshold - hparams.stability_score_offset;

			int ne0 = (int)state.low_res_masks->ne[0];
			int ne1 = (int)state.low_res_masks->ne[1];
			int ne2 = (int)state.low_res_masks->ne[2];

			// Remove padding and upscale masks to the original image size.
			// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

			float preprocess_scale = MathF.Max(nx, ny) / (float)(n_img_size);
			int cropped_nx = (int)(nx / preprocess_scale + 0.5f);
			int cropped_ny = (int)(ny / preprocess_scale + 0.5f);

			float scale_x_1 = (float)ne0 / (float)n_img_size;
			float scale_y_1 = (float)ne1 / (float)n_img_size;

			float scale_x_2 = (float)(cropped_nx) / (float)(nx);
			float scale_y_2 = (float)(cropped_ny) / (float)(ny);

			float* iou_data = (float*)state.iou_predictions->data;

			for (int i = 0; i < ne2; ++i)
			{
				if (iou_threshold > 0.0f && iou_data[i] < iou_threshold)
				{
					Console.WriteLine($"Skipping mask {i} with iou {iou_data[i]} below threshold {iou_threshold}");
					continue; // Filtering masks with iou below the threshold
				}

				float[] mask_data = new float[n_img_size * n_img_size];
				{
					float* data = (float*)state.low_res_masks->data + i * ne0 * ne1;

					for (int iy = 0; iy < n_img_size; ++iy)
					{
						for (int ix = 0; ix < n_img_size; ++ix)
						{
							float sx = MathF.Max(scale_x_1 * (ix + 0.5f) - 0.5f, 0.0f);
							float sy = MathF.Max(scale_y_1 * (iy + 0.5f) - 0.5f, 0.0f);

							int x0 = Math.Max(0, (int)sx);
							int y0 = Math.Max(0, (int)sy);

							int x1 = Math.Min(x0 + 1, (int)ne0 - 1);
							int y1 = Math.Min(y0 + 1, (int)ne1 - 1);

							float dx = sx - x0;
							float dy = sy - y0;

							int j00 = y0 * ne0 + x0;
							int j01 = y0 * ne0 + x1;
							int j10 = y1 * ne0 + x0;
							int j11 = y1 * ne0 + x1;

							float v00 = data[j00];
							float v01 = data[j01];
							float v10 = data[j10];
							float v11 = data[j11];

							float v0 = (1 - dx) * v00 + dx * v01;
							float v1 = (1 - dx) * v10 + dx * v11;

							float v = (1 - dy) * v0 + dy * v1;

							mask_data[iy * n_img_size + ix] = v;
						}
					}
				}

				int intersections = 0;
				int unions = 0;
				sam_image_u8 res;
				int min_iy = ny;
				int max_iy = 0;
				int min_ix = nx;
				int max_ix = 0;
				{
					float[] data = mask_data;

					res.nx = nx;
					res.ny = ny;
					res.data = new byte[nx * ny];

					for (int iy = 0; iy < ny; ++iy)
					{
						for (int ix = 0; ix < nx; ++ix)
						{
							float sx = MathF.Max(scale_x_2 * (ix + 0.5f) - 0.5f, 0.0f);
							float sy = MathF.Max(scale_y_2 * (iy + 0.5f) - 0.5f, 0.0f);

							int x0 = Math.Max(0, (int)sx);
							int y0 = Math.Max(0, (int)sy);

							int x1 = Math.Min(x0 + 1, cropped_nx - 1);
							int y1 = Math.Min(y0 + 1, cropped_ny - 1);

							float dx = sx - x0;
							float dy = sy - y0;

							int j00 = y0 * n_img_size + x0;
							int j01 = y0 * n_img_size + x1;
							int j10 = y1 * n_img_size + x0;
							int j11 = y1 * n_img_size + x1;

							float v00 = data[j00];
							float v01 = data[j01];
							float v10 = data[j10];
							float v11 = data[j11];

							float v0 = (1 - dx) * v00 + dx * v01;
							float v1 = (1 - dx) * v10 + dx * v11;

							float v = (1 - dy) * v0 + dy * v1;

							if (v > intersection_threshold)
							{
								intersections++;
							}
							if (v > union_threshold)
							{
								unions++;
							}
							if (v > mask_threshold)
							{
								min_iy = Math.Min(min_iy, iy);
								max_iy = Math.Max(max_iy, iy);
								min_ix = Math.Min(min_ix, ix);
								max_ix = Math.Max(max_ix, ix);

								res.data[iy * nx + ix] = 255;
							}
						}
					}
				}

				float stability_score = (float)(intersections) / (float)(unions);
				if (stability_score_threshold > 0.0f && stability_score < stability_score_threshold)
				{
					Console.WriteLine("Skipping mask {0} with stability score {1} below threshold {2}", i, stability_score, stability_score_threshold);
					continue; // Filtering masks with stability score below the threshold
				}

				Console.WriteLine("Mask {0}: iou = {1}, stability_score = {2}, bbox ({3}, {4}), ({5}, {6})\n",
						i, iou_data[i], stability_score, min_ix, max_ix, min_iy, max_iy);

				string filename = fname + i + ".png";

				Bitmap bitmap = new Bitmap(res.nx, res.ny, PixelFormat.Format8bppIndexed);
				BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, res.nx, res.ny), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
				Marshal.Copy(res.data, 0, bitmapData.Scan0, res.data.Length);
				bitmap.UnlockBits(bitmapData);
				bitmap.Save(filename, ImageFormat.Png);
			}


			return true;
		}

	}
}

