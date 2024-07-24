using GGMLSharp;
using ModelLoader;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace SAM
{
	internal class Program
	{
		private static Custom1OpDelegate opSin = new Custom1OpDelegate(SamSin);
		private static Custom1OpDelegate opCos = new Custom1OpDelegate(SamCos);
		static void Main(string[] args)
		{
			// First you shold download the sam_vit_b ModelPath and move it to Assets folder.
			// The download link is: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

			SamParams samParam = new SamParams()
			{
				IouThreshold = 0.9f,
				StabilityScoreThreshold = 0.9f,
				Point = new PointF(248.0f, 162.0f),
				ModelPath = @"./Assets/sam_vit_b_01ec64.pth",
				ImageInputPath = @"./Assets/example.jpg",
				Threads = 16,
			};

			// Load ModelPath
			SamModel model = LoadModel(samParam);
			Console.WriteLine("Model Loaded.");

			Console.WriteLine("Load image from file...");
			// Load Img
			SamImageU8 imgU8 = LoadImageFromFile(samParam.ImageInputPath);
			SamImageF32 imgF32 = PreprocessImage(imgU8);

			Console.WriteLine("Init state");
			// Init state
			SamState state = new SamState();
			{
				ulong bufSize = 256u * 1024 * 1024;
				state.context = new SafeGGmlContext(IntPtr.Zero, bufSize);
				state.embdImg = state.context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, model.hparams.ImgEmbdCount, model.hparams.ImgEmbdCount, model.hparams.EncoderOutChans);
				state.lowResMasks = state.context.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, model.hparams.EncoderOutChans, model.hparams.EncoderOutChans, 3);
				state.iouPredictions = state.context.NewTensor1d(Structs.GGmlType.GGML_TYPE_F32, 3);
			}

			// Encode image
			{
				Console.WriteLine("Encoding the image...");
				state.allocr = new SafeGGmlGraphAllocr();

				SafeGGmlGraph gf = SamEncodeImage(model, state, imgF32);
				if (gf.IsInvalid)
				{
					throw new Exception("failed to encode image");
				}

				GraphComputeHelper(gf, samParam.Threads);
				state.allocr.Free();
				Console.WriteLine("Image encoded done.");
			}

			// Compute the masks
			{
				state.allocr = new SafeGGmlGraphAllocr();

				// TODO: more varied prompts
				Console.WriteLine($"prompt Point: ({samParam.Point.X}, {samParam.Point.Y})");
				Console.WriteLine("Computing the mask...");
				SafeGGmlGraph graph = SamBuildFastGraph(model, state, imgU8.Width, imgU8.Height, samParam.Point);

				if (graph.IsInvalid)
				{
					throw new Exception("failed to build fast graph");
				}

				GraphComputeHelper(graph, samParam.Threads);

				state.allocr.Free();
				Console.WriteLine("Masks have got.");
			}


			// Write masks.

			Console.WriteLine("Writing masks...");
			WriteMasks(model.hparams, imgU8.Width, imgU8.Height, state, samParam.OutputNameHeader);

			Console.WriteLine("Write masks done.");
			Console.ReadKey();
		}

		class SamHparams
		{
			public int EncoderState = 768;
			public int EncoderLayer = 12;
			public int EncoderHead = 12;
			public int EncoderOutChans = 256;
			public int PtEmbd = 4;
			public int DecoderHeadsCount = 8;
			public int Type = 1;
			public float MaskThreshold = 0.0f;
			public float IouThreshold = 0.88f;
			public float StabilityScoreThreshold = 0.95f;
			public float StabilityScoreOffset = 1.0f;
			public float Eps = 1e-6f;
			public float EpsDecoderTransformer = 1e-5f;

			public int EncHeadDim => EncoderState / EncoderHead;
			public int ImgSize => 1024;
			public int WindowSize => 14;
			public int PatchSize => 16;
			public int ImgEmbdCount => ImgSize / PatchSize;

			public int[] GlobalAttnIndices
			{
				get
				{
					switch (EncoderState)
					{
						case 768: return new int[] { 2, 5, 8, 11 };
						case 1024: return new int[] { 5, 11, 17, 23 };
						case 1280: return new int[] { 7, 15, 23, 31 };
						default:
							{
								throw new ArgumentOutOfRangeException($"unsupported EncoderState = {EncoderState}");
							}
					};
				}
			}

			public bool IsGlobalAttn(int layer)
			{
				return GlobalAttnIndices.Contains(layer);
			}
		};


		// RGB uint8 image
		struct SamImageU8
		{
			public int Width;
			public int Height;

			public byte[] Data;
		};

		// RGB float32 image
		// Memory layout: RGBRGBRGB...
		struct SamImageF32
		{
			public int Width;
			public int Height;

			public float[] Data;
		};

		class SamModel
		{
			public SamHparams hparams = new SamHparams();
			public SamEncoderImage encoderImg;
			public SamEncoderPrompt encoderPrompt;
			public SamDecoderMask decoderMask;
			public SafeGGmlContext context;
		};

		struct SamLayerEnc
		{
			public SafeGGmlTensor norm1_w;
			public SafeGGmlTensor norm1_b;

			public SafeGGmlTensor rel_pos_w;
			public SafeGGmlTensor rel_pos_h;

			public SafeGGmlTensor qkv_w;
			public SafeGGmlTensor qkv_b;

			public SafeGGmlTensor proj_w;
			public SafeGGmlTensor proj_b;

			public SafeGGmlTensor norm2_w;
			public SafeGGmlTensor norm2_b;

			public SafeGGmlTensor mlp_lin1_w;
			public SafeGGmlTensor mlp_lin1_b;

			public SafeGGmlTensor mlp_lin2_w;
			public SafeGGmlTensor mlp_lin2_b;
		};

		struct SamEncoderImage
		{
			public SafeGGmlTensor pe;

			public SafeGGmlTensor proj_w;
			public SafeGGmlTensor proj_b;

			public SafeGGmlTensor neck_conv_0;
			public SafeGGmlTensor neck_norm_0_w;
			public SafeGGmlTensor neck_norm_0_b;
			public SafeGGmlTensor neck_conv_1;
			public SafeGGmlTensor neck_norm_1_w;
			public SafeGGmlTensor neck_norm_1_b;

			public SamLayerEnc[] layers;
		};

		struct SamEncoderPrompt
		{
			public SafeGGmlTensor pe;

			public SafeGGmlTensor not_a_pt_embd_w;
			public SafeGGmlTensor[] pt_embd;

			public SafeGGmlTensor no_mask_embd_w;
		};

		struct SamLayerDecTransformerAttn
		{
			// q_proj
			public SafeGGmlTensor q_w;
			public SafeGGmlTensor q_b;

			// k_proj
			public SafeGGmlTensor k_w;
			public SafeGGmlTensor k_b;

			// v_proj
			public SafeGGmlTensor v_w;
			public SafeGGmlTensor v_b;

			// out_proj
			public SafeGGmlTensor out_w;
			public SafeGGmlTensor out_b;
		};

		struct SamLayerDecTransformer
		{
			public SamLayerDecTransformerAttn self_attn;

			// norm1
			public SafeGGmlTensor norm1_w;
			public SafeGGmlTensor norm1_b;

			public SamLayerDecTransformerAttn cross_attn_token_to_img;

			// norm2
			public SafeGGmlTensor norm2_w;
			public SafeGGmlTensor norm2_b;

			// mlp.lin1
			public SafeGGmlTensor mlp_lin1_w;
			public SafeGGmlTensor mlp_lin1_b;

			// mlp.lin2
			public SafeGGmlTensor mlp_lin2_w;
			public SafeGGmlTensor mlp_lin2_b;

			// norm3
			public SafeGGmlTensor norm3_w;
			public SafeGGmlTensor norm3_b;

			// norm4
			public SafeGGmlTensor norm4_w;
			public SafeGGmlTensor norm4_b;

			public SamLayerDecTransformerAttn cross_attn_img_to_token;
		};

		struct SamLayerDecOutputHypernetMlps
		{
			// mlps_*.layers.0
			public SafeGGmlTensor w_0;
			public SafeGGmlTensor b_0;

			// mlps_*.layers.1
			public SafeGGmlTensor w_1;
			public SafeGGmlTensor b_1;

			// mlps_*.layers.2
			public SafeGGmlTensor w_2;
			public SafeGGmlTensor b_2;
		};

		struct SamDecoderMask
		{
			public SamLayerDecTransformer[] transformer_layers;

			// trasnformer.final_attn_token_to_image
			public SamLayerDecTransformerAttn transformer_final_attn_token_to_img;

			// transformer.norm_final
			public SafeGGmlTensor transformer_norm_final_w;
			public SafeGGmlTensor transformer_norm_final_b;

			// output_upscaling.0
			public SafeGGmlTensor output_upscaling_0_w;
			public SafeGGmlTensor output_upscaling_0_b;

			// output_upscaling.1
			public SafeGGmlTensor output_upscaling_1_w;
			public SafeGGmlTensor output_upscaling_1_b;

			// output_upscaling.3
			public SafeGGmlTensor output_upscaling_3_w;
			public SafeGGmlTensor output_upscaling_3_b;

			// output_hypernetworks_mlps
			public SamLayerDecOutputHypernetMlps[] output_hypernet_mlps;

			// iou_prediction_head.0
			public SafeGGmlTensor iou_prediction_head_0_w;
			public SafeGGmlTensor iou_prediction_head_0_b;

			// iou_prediction_head.1
			public SafeGGmlTensor iou_prediction_head_1_w;
			public SafeGGmlTensor iou_prediction_head_1_b;

			// iou_prediction_head.2
			public SafeGGmlTensor iou_prediction_head_2_w;
			public SafeGGmlTensor iou_prediction_head_2_b;

			// iou_token.weight
			public SafeGGmlTensor iou_token_w;

			// mask_tokens.weight
			public SafeGGmlTensor mask_tokens_w;
		};

		struct SamState
		{
			public SafeGGmlTensor embdImg;

			public SafeGGmlTensor lowResMasks;
			public SafeGGmlTensor iouPredictions;

			//  ggml_tensor * tmp_save = {};

			public SafeGGmlContext context;

			public SafeGGmlGraphAllocr allocr;
		};

		class SamParams
		{
			public int Seed = -1; // RNG Seed
			public int Threads = 4;

			public string ModelPath = "./Assets/sam_vit_b_01ec64.pth"; // ModelPath path
			public string ImageInputPath = "./Assets/example.jpg";
			public string OutputNameHeader = "img.out";
			public float MaskThreshold = 0.0f;
			public float IouThreshold = 0.88f;
			public float StabilityScoreThreshold = 0.95f;
			public float StabilityScoreOffset = 1.0f;
			public float Eps = 1e-6f;
			public float EpsDecoderTransformer = 1e-5f;
			public PointF Point = new PointF(414.375f, 162.796875f);
		};

		static void DisconnectNodeFromGraph(SafeGGmlTensor t)
		{
			t.Operations = Structs.GGmlOperation.GGML_OP_NONE;
			for (int i = 0; i < Structs.GGML_MAX_SRC; i++)
			{
				t.Sources[i].Dispose();
			}
		}

		static void GraphComputeHelper(SafeGGmlGraph graph, int threads)
		{
			ulong mem_size = Common.TensorOverheadLength * Structs.GGML_DEFAULT_GRAPH_SIZE * (ulong)graph.NodeCount + Common.GraphOverheadLength;
			SafeGGmlContext context = new SafeGGmlContext(IntPtr.Zero, mem_size, false);
			graph.ComputeWithGGmlContext(context, threads);
		}

		static void SamSin(SafeGGmlTensor dst, SafeGGmlTensor src, int ith, int nth, IntPtr userdata)
		{
			if (userdata != IntPtr.Zero)
			{
				throw new Exception("userdata is not null");
			}
			if (!dst.AreSameShape(src))
			{
				throw new Exception("dst and src are not the same shape");
			}
			if (!dst.IsContiguous())
			{
				throw new Exception("des is not contiguous");
			}
			if (!src.IsContiguous())
			{
				throw new Exception("src is not contiguous");
			}

			float[] src_data = GGMLSharp.DataConverter.ConvertToFloats(src.GetData());

			int ne = (int)dst.ElementsCount;
			int dr = (ne + nth - 1) / nth;
			int ie0 = dr * ith;
			int ie1 = Math.Min(ie0 + dr, ne);

			for (int i = ie0; i < ie1; ++i)
			{
				dst.SetFloat(i, (float)Math.Sin(src_data[i]));
			}

		}

		static void SamCos(SafeGGmlTensor dst, SafeGGmlTensor src, int ith, int nth, IntPtr userdata)
		{
			if (userdata != IntPtr.Zero)
			{
				throw new Exception("userdata is not null");
			}
			if (!dst.AreSameShape(src))
			{
				throw new Exception("dst and src are not the same shape");
			}
			if (!dst.IsContiguous())
			{
				throw new Exception("des is not contiguous");
			}
			if (!src.IsContiguous())
			{
				throw new Exception("src is not contiguous");
			}

			float[] src_data = GGMLSharp.DataConverter.ConvertToFloats(src.GetData());

			int ne = (int)dst.ElementsCount;
			int dr = (ne + nth - 1) / nth;
			int ie0 = dr * ith;
			int ie1 = Math.Min(ie0 + dr, ne);

			for (int i = ie0; i < ie1; ++i)
			{
				dst.SetFloat(i, (float)Math.Cos(src_data[i]));
			}
		}

		static SamImageU8 LoadImageFromFile(string fname)
		{
			Bitmap bitmap = new Bitmap(fname);
			SamImageU8 img = new SamImageU8();
			img.Width = bitmap.Width;
			img.Height = bitmap.Height;
			img.Data = new byte[img.Width * img.Height * 3];

			BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
			Marshal.Copy(bitmapData.Scan0, img.Data, 0, img.Data.Length);
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
		static SamImageF32 PreprocessImage(SamImageU8 img)
		{
			SamImageF32 res = new SamImageF32();
			int nx = img.Width;
			int ny = img.Height;

			int nx2 = 1024;
			int ny2 = 1024;

			res.Width = nx2;
			res.Height = ny2;
			res.Data = new float[3 * nx2 * ny2];

			float scale = (float)Math.Max(nx, ny) / 1024.0f;

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

						float v00 = img.Data[j00];
						float v01 = img.Data[j01];
						float v10 = img.Data[j10];
						float v11 = img.Data[j11];

						float v0 = v00 * (1.0f - dx) + v01 * dx;
						float v1 = v10 * (1.0f - dx) + v11 * dx;

						float v = v0 * (1.0f - dy) + v1 * dy;

						double v2 = Math.Min(Math.Max(Math.Round(v), 0.0f), 255.0f);

						int i = 3 * (y * nx3 + x) + c;

						res.Data[i] = ((float)(v2) - m3[c]) / s3[c];
					}
				}
			}

			return res;
		}

		static SamModel LoadModel(SamParams samParams)
		{
			if (!File.Exists(samParams.ModelPath))
			{
				throw new FileNotFoundException("Model file not found");
			}

			PickleLoader modelLoader = new PickleLoader();
			Console.WriteLine($"loading ModelPath from {samParams.ModelPath} - please wait ...");
			List<Tensor> tensors = modelLoader.ReadTensorsInfoFromFile(samParams.ModelPath);
			Console.WriteLine($"ModelPath header loaded, total layers is: {tensors.Count}");

			ulong contextSize = 0;
			tensors.ForEach(a =>
			{
				ulong tensorSize = Common.GetGGmlTypeSize(a.Type);
				a.Shape.ForEach(ne =>
				{
					tensorSize *= (ulong)ne;
				});
				contextSize += tensorSize;
			});
			contextSize += Common.TensorOverheadLength * (ulong)tensors.Count;

			Console.WriteLine("Total context size: " + (float)contextSize / 1024 / 1024 + " MB");

			SafeGGmlContext context = new SafeGGmlContext(IntPtr.Zero, contextSize, false);

			SamModel model = new SamModel();
			model.context = context;

			//SamHparams hparams = ModelPath.hparams;
			SamHparams hparams = new SamHparams
			{
				Eps = samParams.Eps,
				EpsDecoderTransformer = samParams.EpsDecoderTransformer,
				IouThreshold = samParams.IouThreshold,
				MaskThreshold = samParams.MaskThreshold,
				StabilityScoreOffset = samParams.StabilityScoreOffset,
				StabilityScoreThreshold = samParams.StabilityScoreThreshold,
			};

			model.hparams = hparams;

			int encState = hparams.EncoderState;
			int encLayer = hparams.EncoderLayer;
			int encHeadDim = hparams.EncHeadDim;
			int encoderOutChans = hparams.EncoderOutChans;
			int ptEmbd = hparams.PtEmbd;

			int imgEmbdCount = hparams.ImgEmbdCount;
			int windowSize = hparams.WindowSize;
			int patchSize = hparams.PatchSize;


			// image encoder
			{
				model.encoderImg.pe = GetTensorFromName(context, modelLoader, tensors, "image_encoder.pos_embed", new long[] { encState, imgEmbdCount, imgEmbdCount });

				model.encoderImg.proj_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.patch_embed.proj.weight", new long[] { patchSize, patchSize, 3, encState }, Structs.GGmlType.GGML_TYPE_F16);
				model.encoderImg.proj_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.patch_embed.proj.bias", new long[] { 1, 1, encState });

				model.encoderImg.neck_conv_0 = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.0.weight", new long[] { 1, 1, encState, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
				model.encoderImg.neck_conv_1 = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.2.weight", new long[] { 3, 3, encoderOutChans, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);

				model.encoderImg.neck_norm_0_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.1.weight");
				model.encoderImg.neck_norm_0_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.1.bias");

				model.encoderImg.neck_norm_1_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.3.weight");
				model.encoderImg.neck_norm_1_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.neck.3.bias");

				model.encoderImg.layers = new SamLayerEnc[model.hparams.EncoderLayer];

				for (int i = 0; i < model.hparams.EncoderLayer; ++i)
				{
					model.encoderImg.layers[i].norm1_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".norm1.weight");
					model.encoderImg.layers[i].norm1_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".norm1.bias");
					if (hparams.IsGlobalAttn(i))
					{
						model.encoderImg.layers[i].rel_pos_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_w", new long[] { encHeadDim, 2 * imgEmbdCount - 1 }, Structs.GGmlType.GGML_TYPE_F16);
						model.encoderImg.layers[i].rel_pos_h = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_h", new long[] { encHeadDim, 2 * imgEmbdCount - 1 }, Structs.GGmlType.GGML_TYPE_F16);
					}
					else
					{
						model.encoderImg.layers[i].rel_pos_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_w", new long[] { encHeadDim, 2 * windowSize - 1 }, Structs.GGmlType.GGML_TYPE_F16);
						model.encoderImg.layers[i].rel_pos_h = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.rel_pos_h", new long[] { encHeadDim, 2 * windowSize - 1 }, Structs.GGmlType.GGML_TYPE_F16);
					}

					model.encoderImg.layers[i].qkv_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.qkv.weight", new long[] { encState, 3 * encState, 1, 1 }, Structs.GGmlType.GGML_TYPE_F16);
					model.encoderImg.layers[i].qkv_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.qkv.bias");
					model.encoderImg.layers[i].proj_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.proj.weight");
					model.encoderImg.layers[i].proj_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".attn.proj.bias");
					model.encoderImg.layers[i].norm2_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".norm2.weight");
					model.encoderImg.layers[i].norm2_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".norm2.bias");
					model.encoderImg.layers[i].mlp_lin1_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".mlp.lin1.weight", new long[] { encState, 4 * encState }, Structs.GGmlType.GGML_TYPE_F16);
					model.encoderImg.layers[i].mlp_lin1_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".mlp.lin1.bias");
					model.encoderImg.layers[i].mlp_lin2_w = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".mlp.lin2.weight", new long[] { 4 * encState, encState }, Structs.GGmlType.GGML_TYPE_F16);
					model.encoderImg.layers[i].mlp_lin2_b = GetTensorFromName(context, modelLoader, tensors, "image_encoder.blocks." + i + ".mlp.lin2.bias");
				}
			}

			// prompt encoder
			{
				model.encoderPrompt.pe = GetTensorFromName(context, modelLoader, tensors, "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", new long[] { encoderOutChans / 2, 2 }, Structs.GGmlType.GGML_TYPE_F32);
				model.encoderPrompt.not_a_pt_embd_w = GetTensorFromName(context, modelLoader, tensors, "prompt_encoder.not_a_point_embed.weight", new long[] { encoderOutChans, 1 });
				model.encoderPrompt.no_mask_embd_w = GetTensorFromName(context, modelLoader, tensors, "prompt_encoder.no_mask_embed.weight", new long[] { encoderOutChans, 1 });
				model.encoderPrompt.pt_embd = new SafeGGmlTensor[model.hparams.PtEmbd];
				for (int i = 0; i < model.hparams.PtEmbd; i++)
				{
					model.encoderPrompt.pt_embd[i] = GetTensorFromName(context, modelLoader, tensors, "prompt_encoder.point_embeddings." + i + ".weight", new long[] { encoderOutChans, 1 });
				}
			}

			// mask decoder
			{
				int tfm_layers_count = 2;
				model.decoderMask.transformer_layers = new SamLayerDecTransformer[tfm_layers_count];
				for (int i = 0; i < tfm_layers_count; ++i)
				{
					string prefix = "mask_decoder.transformer.layers." + i + ".";
					model.decoderMask.transformer_layers[i].self_attn.q_w = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.q_proj.weight");
					model.decoderMask.transformer_layers[i].self_attn.q_b = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.q_proj.bias");
					model.decoderMask.transformer_layers[i].self_attn.k_w = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.k_proj.weight");
					model.decoderMask.transformer_layers[i].self_attn.k_b = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.k_proj.bias");
					model.decoderMask.transformer_layers[i].self_attn.v_w = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.v_proj.weight");
					model.decoderMask.transformer_layers[i].self_attn.v_b = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.v_proj.bias");
					model.decoderMask.transformer_layers[i].self_attn.out_w = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.out_proj.weight");
					model.decoderMask.transformer_layers[i].self_attn.out_b = GetTensorFromName(context, modelLoader, tensors, prefix + "self_attn.out_proj.bias");
					model.decoderMask.transformer_layers[i].norm1_w = GetTensorFromName(context, modelLoader, tensors, prefix + "norm1.weight");
					model.decoderMask.transformer_layers[i].norm1_b = GetTensorFromName(context, modelLoader, tensors, prefix + "norm1.bias");
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.q_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.q_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.q_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.q_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.k_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.k_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.k_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.k_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.v_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.v_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.v_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.v_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.out_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.out_proj.weight", new long[] { encoderOutChans / 2, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_token_to_img.out_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_token_to_image.out_proj.bias");
					model.decoderMask.transformer_layers[i].norm2_w = GetTensorFromName(context, modelLoader, tensors, prefix + "norm2.weight");
					model.decoderMask.transformer_layers[i].norm2_b = GetTensorFromName(context, modelLoader, tensors, prefix + "norm2.bias");
					model.decoderMask.transformer_layers[i].mlp_lin1_w = GetTensorFromName(context, modelLoader, tensors, prefix + "mlp.lin1.weight", new long[] { encoderOutChans, 8 * encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].mlp_lin1_b = GetTensorFromName(context, modelLoader, tensors, prefix + "mlp.lin1.bias");
					model.decoderMask.transformer_layers[i].mlp_lin2_w = GetTensorFromName(context, modelLoader, tensors, prefix + "mlp.lin2.weight", new long[] { 8 * encoderOutChans, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].mlp_lin2_b = GetTensorFromName(context, modelLoader, tensors, prefix + "mlp.lin2.bias");
					model.decoderMask.transformer_layers[i].norm3_w = GetTensorFromName(context, modelLoader, tensors, prefix + "norm3.weight");
					model.decoderMask.transformer_layers[i].norm3_b = GetTensorFromName(context, modelLoader, tensors, prefix + "norm3.bias");
					model.decoderMask.transformer_layers[i].norm4_w = GetTensorFromName(context, modelLoader, tensors, prefix + "norm4.weight");
					model.decoderMask.transformer_layers[i].norm4_b = GetTensorFromName(context, modelLoader, tensors, prefix + "norm4.bias");
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.q_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.q_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.q_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.q_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.k_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.k_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.k_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.k_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.v_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.v_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.v_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.v_proj.bias");
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.out_w = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.out_proj.weight", new long[] { encoderOutChans / 2, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
					model.decoderMask.transformer_layers[i].cross_attn_img_to_token.out_b = GetTensorFromName(context, modelLoader, tensors, prefix + "cross_attn_image_to_token.out_proj.bias");
				}

				model.decoderMask.transformer_final_attn_token_to_img.q_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.q_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.transformer_final_attn_token_to_img.q_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.q_proj.bias");
				model.decoderMask.transformer_final_attn_token_to_img.k_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.k_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.transformer_final_attn_token_to_img.k_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.k_proj.bias");
				model.decoderMask.transformer_final_attn_token_to_img.v_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.v_proj.weight", new long[] { encoderOutChans, encoderOutChans / 2 }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.transformer_final_attn_token_to_img.v_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.v_proj.bias");
				model.decoderMask.transformer_final_attn_token_to_img.out_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.out_proj.weight", new long[] { encoderOutChans / 2, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.transformer_final_attn_token_to_img.out_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.final_attn_token_to_image.out_proj.bias");

				model.decoderMask.transformer_norm_final_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.norm_final_attn.weight");
				model.decoderMask.transformer_norm_final_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.transformer.norm_final_attn.bias");

				model.decoderMask.output_upscaling_0_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.0.weight", new long[] { 2, 2, imgEmbdCount, encoderOutChans }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.output_upscaling_0_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.0.bias");
				model.decoderMask.output_upscaling_1_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.1.weight");
				model.decoderMask.output_upscaling_1_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.1.bias");
				model.decoderMask.output_upscaling_3_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.3.weight", new long[] { 2, 2, imgEmbdCount / 2, imgEmbdCount }, Structs.GGmlType.GGML_TYPE_F16);
				model.decoderMask.output_upscaling_3_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.output_upscaling.3.bias");

				int n_hypernet_mpls_count = 4;
				model.decoderMask.output_hypernet_mlps = new SamLayerDecOutputHypernetMlps[n_hypernet_mpls_count];
				for (int i = 0; i < n_hypernet_mpls_count; ++i)
				{
					string prefix = "mask_decoder.output_hypernetworks_mlps." + i + ".";
					model.decoderMask.output_hypernet_mlps[i].w_0 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.0.weight");
					model.decoderMask.output_hypernet_mlps[i].b_0 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.0.bias");
					model.decoderMask.output_hypernet_mlps[i].w_1 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.1.weight");
					model.decoderMask.output_hypernet_mlps[i].b_1 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.1.bias");
					model.decoderMask.output_hypernet_mlps[i].w_2 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.2.weight", new long[] { encoderOutChans, imgEmbdCount / 2 });
					model.decoderMask.output_hypernet_mlps[i].b_2 = GetTensorFromName(context, modelLoader, tensors, prefix + "layers.2.bias");
				}

				model.decoderMask.iou_prediction_head_0_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.0.weight");
				model.decoderMask.iou_prediction_head_0_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.0.bias");
				model.decoderMask.iou_prediction_head_1_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.1.weight");
				model.decoderMask.iou_prediction_head_1_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.1.bias");
				model.decoderMask.iou_prediction_head_2_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.2.weight", new long[] { encoderOutChans, ptEmbd });
				model.decoderMask.iou_prediction_head_2_b = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_prediction_head.layers.2.bias");

				model.decoderMask.iou_token_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.iou_token.weight", new long[] { encoderOutChans, 1 });
				model.decoderMask.mask_tokens_w = GetTensorFromName(context, modelLoader, tensors, "mask_decoder.mask_tokens.weight", new long[] { encoderOutChans, ptEmbd });

			}
			return model;
		}

		static SafeGGmlTensor GetTensorFromName(SafeGGmlContext context, PickleLoader loader, List<Tensor> tensors, string name, Structs.GGmlType type = Structs.GGmlType.GGML_TYPE_F32)
		{
			var tensor = tensors.Find(x => x.Name == name);
			if (tensor == null)
			{
				throw new Exception($"tensor {name} not found");
			}
			byte[] bytes = loader.ReadByteFromFile(tensor);
			bytes = TransData(bytes, tensor.Type, type);
			long[] ne = new long[tensor.Shape.Count];
			for (int i = 0; i < tensor.Shape.Count; i++)
			{
				ne[i] = (long)tensor.Shape[i];
			}
			SafeGGmlTensor t = context.NewTensor(type, ne);
			t.SetData(bytes);
			t.Name = tensor.Name;
			return t;
		}

		static SafeGGmlTensor GetTensorFromName(SafeGGmlContext context, ModelLoader.PickleLoader loader, List<ModelLoader.Tensor> tensors, string name, long[] shape, Structs.GGmlType type = Structs.GGmlType.GGML_TYPE_F32)
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
			byte[] bytes = loader.ReadByteFromFile(tensor);
			bytes = TransData(bytes, tensor.Type, type);
			SafeGGmlTensor t = context.NewTensor(type, shape);
			t.SetData(bytes);
			t.Name = tensor.Name;
			return t;
		}

		static byte[] TransData(byte[] data, Structs.GGmlType srcType, Structs.GGmlType desType)
		{
			if (srcType == desType)
			{
				return data;
			}
			if (srcType != Structs.GGmlType.GGML_TYPE_F16 && srcType != Structs.GGmlType.GGML_TYPE_BF16 && srcType != Structs.GGmlType.GGML_TYPE_F32)
			{
				throw new ArgumentException("Src Type not support");
			}
			if (desType != Structs.GGmlType.GGML_TYPE_F16 && desType != Structs.GGmlType.GGML_TYPE_BF16 && desType != Structs.GGmlType.GGML_TYPE_F32)
			{
				throw new ArgumentException("Des Type not support");
			}

			if (srcType == Structs.GGmlType.GGML_TYPE_BF16)
			{
				if (desType == Structs.GGmlType.GGML_TYPE_F16)
				{
					GGMLSharp.DataConverter.Bf16ToFp16Bytes(data);
					return data;
				}
				else if (desType == Structs.GGmlType.GGML_TYPE_F32)
				{
					GGMLSharp.DataConverter.Bf16ToFp32Bytes(ref data);
					return data;
				}
			}
			else if (srcType == Structs.GGmlType.GGML_TYPE_F32)
			{
				if (desType == Structs.GGmlType.GGML_TYPE_BF16)
				{
					GGMLSharp.DataConverter.Fp32ToBf16Bytes(data);
				}
				else if (desType == Structs.GGmlType.GGML_TYPE_F16)
				{
					GGMLSharp.DataConverter.Fp32ToFp16Bytes(data);
				}
				return data;
			}
			else if (srcType == Structs.GGmlType.GGML_TYPE_F16)
			{
				if (desType == Structs.GGmlType.GGML_TYPE_F32)
				{
					GGMLSharp.DataConverter.Fp16ToFp32Bytes(ref data);
					return data;
				}
			}
			throw new ArgumentException("Not support!");
		}

		struct PromptEncoderResult
		{
			public SafeGGmlTensor embdPromptSparse;
			public SafeGGmlTensor embdPromptDense;
		};

		static SafeGGmlTensor SamFillDensePe(SamModel model, SafeGGmlContext ctx0, SafeGGmlGraph gf, SamState state)
		{
			SamHparams hparams = model.hparams;
			SamEncoderPrompt enc = model.encoderPrompt;

			int n_img_embd = hparams.ImgEmbdCount;
			SafeGGmlTensor xy_embed_stacked = ctx0.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, 2, n_img_embd, n_img_embd);
			xy_embed_stacked.Name = "xy_embed_stacked";
			xy_embed_stacked.SetInput();

			SafeGGmlTensor cur = ctx0.MulMat(ctx0.Cont(ctx0.Transpose(enc.pe)), xy_embed_stacked);

			cur = ctx0.Scale(cur, 2.0f * (float)Math.PI);

			// concat
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
			{
				SafeGGmlTensor t_sin = ctx0.MapCustom1(cur, opSin, GGML_N_TASKS_MAX, IntPtr.Zero);
				SafeGGmlTensor t_cos = ctx0.MapCustom1(cur, opCos, GGML_N_TASKS_MAX, IntPtr.Zero);

				cur = ctx0.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, t_sin.Shape[0] + t_cos.Shape[0], cur.Shape[1], cur.Shape[2]);

				gf.BuildForwardExpend(ctx0.Copy(t_sin, ctx0.View3d(cur, t_sin.Shape[0], t_sin.Shape[1], t_sin.Shape[2], cur.Stride[1], cur.Stride[2], 0)));
				gf.BuildForwardExpend(ctx0.Copy(t_cos, ctx0.View3d(cur, t_sin.Shape[0], t_sin.Shape[1], t_sin.Shape[2], cur.Stride[1], cur.Stride[2], t_sin.Stride[1])));
			}

			SafeGGmlTensor pe_img_dense = ctx0.Cont(ctx0.Permute(cur, 2, 0, 1, 3));
			gf.BuildForwardExpend(pe_img_dense);

			return pe_img_dense;
		}

		static PromptEncoderResult SamEncodePrompt(SamModel model, SafeGGmlContext ctx0, SafeGGmlGraph gf, SamState state)
		{
			SamHparams hparams = model.hparams;
			SamEncoderPrompt enc = model.encoderPrompt;

			SafeGGmlTensor inp = ctx0.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, 2, 2);
			inp.Name = "prompt_input";
			inp.SetInput();


			SafeGGmlTensor cur = ctx0.MulMat(ctx0.Cont(ctx0.Transpose(enc.pe)), inp);

			cur = ctx0.Scale(cur, 2.0f * (float)Math.PI);

			// concat
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
			{
				SafeGGmlTensor t_sin = ctx0.MapCustom1(cur, opSin, GGML_N_TASKS_MAX, IntPtr.Zero);
				SafeGGmlTensor t_cos = ctx0.MapCustom1(cur, opCos, GGML_N_TASKS_MAX, IntPtr.Zero);

				cur = ctx0.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, t_sin.Shape[0] + t_cos.Shape[0], cur.Shape[1]);

				gf.BuildForwardExpend(ctx0.Copy(t_sin, ctx0.View2d(cur, t_sin.Shape[0], t_sin.Shape[1], cur.Stride[1], 0)));
				gf.BuildForwardExpend(ctx0.Copy(t_cos, ctx0.View2d(cur, t_sin.Shape[0], t_sin.Shape[1], cur.Stride[1], t_sin.Stride[1])));

				// overwrite label == -1 with not_a_point_embed.weight
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L86
				// TODO: extend for multiple points
				gf.BuildForwardExpend(ctx0.Copy(enc.not_a_pt_embd_w, ctx0.View2d(cur, cur.Shape[0], 1, cur.Stride[1], cur.Stride[1])));
			}

			// add point_embeddings[1] to label == 1
			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L90
			SafeGGmlTensor v = ctx0.View2d(cur, cur.Shape[0], 1, cur.Stride[1], 0);
			gf.BuildForwardExpend(ctx0.Copy(ctx0.AddInplace(v, enc.pt_embd[1]), v));

			SafeGGmlTensor embd_prompt_sparse = cur;
			gf.BuildForwardExpend(embd_prompt_sparse);

			SafeGGmlTensor embd_prompt_dense = ctx0.Repeat(
				ctx0.Cont(
					ctx0.View3d(enc.no_mask_embd_w,
						  1, 1, enc.no_mask_embd_w.Shape[0], enc.no_mask_embd_w.Stride[0], enc.no_mask_embd_w.Stride[0], 0)),
				ctx0.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, hparams.ImgEmbdCount, hparams.ImgEmbdCount, hparams.EncoderOutChans));

			gf.BuildForwardExpend(embd_prompt_dense);

			//printf("used_mem = %zu\n", ggml_used_mem(ctx0));

			PromptEncoderResult res;
			res.embdPromptSparse = embd_prompt_sparse;
			res.embdPromptDense = embd_prompt_dense;
			return res;
		}

		static SafeGGmlTensor SamDecodeMaskMlpRelu3(SafeGGmlContext ctx0, SafeGGmlTensor input, SafeGGmlTensor w0, SafeGGmlTensor b0, SafeGGmlTensor w1, SafeGGmlTensor b1, SafeGGmlTensor w2, SafeGGmlTensor b2)
		{
			SafeGGmlTensor cur = ctx0.Linear(input, w0, b0);
			cur = ctx0.Relu(cur);
			cur = ctx0.Linear(cur, w1, b1);
			cur = ctx0.Relu(cur);
			cur = ctx0.Linear(cur, w2, b2);
			return cur;
		}

		static bool SamDecodeMask(SamModel model, PromptEncoderResult prompt, SafeGGmlTensor pe_img, SafeGGmlContext ctx0, SafeGGmlGraph gf, SamState state)
		{
			SamHparams hparams = model.hparams;
			SamDecoderMask dec = model.decoderMask;
			int n_img_embd = hparams.ImgEmbdCount;

			SafeGGmlTensor tokens;
			{
				// Concatenate output tokens
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
				SafeGGmlTensor sparse = prompt.embdPromptSparse;

				tokens = ctx0.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, dec.iou_token_w.Shape[0], dec.iou_token_w.Shape[1] + dec.mask_tokens_w.Shape[1] + sparse.Shape[1], sparse.Shape[2]);

				ulong[] offsets = { 0, (ulong)dec.iou_token_w.Shape[1] * tokens.Stride[1], (ulong)dec.iou_token_w.Shape[1] * tokens.Stride[1] + (ulong)dec.mask_tokens_w.Shape[1] * tokens.Stride[1] };
				gf.BuildForwardExpend(ctx0.Copy(dec.iou_token_w, ctx0.View2d(tokens, tokens.Shape[0], dec.iou_token_w.Shape[1], tokens.Stride[1], offsets[0])));
				gf.BuildForwardExpend(ctx0.Copy(dec.mask_tokens_w, ctx0.View2d(tokens, tokens.Shape[0], dec.mask_tokens_w.Shape[1], tokens.Stride[1], offsets[1])));
				gf.BuildForwardExpend(ctx0.Copy(sparse, ctx0.View2d(tokens, tokens.Shape[0], sparse.Shape[1], tokens.Stride[1], offsets[2])));
				// TODO: Sparse prompt embeddings can have more than one Point
			}

			SafeGGmlTensor src;
			SafeGGmlTensor posSrc;
			long[] srcNE = { 0, 0, 0, 0 };
			{
				// Expand per-image Data in the batch direction to be per-mask
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
				src = ctx0.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, state.embdImg.Shape[0], state.embdImg.Shape[1], state.embdImg.Shape[2], tokens.Shape[2]);

				src = ctx0.Add(
					ctx0.Repeat(
						state.embdImg,
						src),
					prompt.embdPromptDense);

				srcNE[0] = src.Shape[0];
				srcNE[1] = src.Shape[1];
				srcNE[2] = src.Shape[2];
				srcNE[3] = src.Shape[3];

				// flatten & permute
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
				src = ctx0.Cont(ctx0.Permute(
					ctx0.View3d(
						src,
						src.Shape[0] * src.Shape[1],
						src.Shape[2],
						src.Shape[3],
						src.Stride[2],
						src.Stride[3],
						0),
					1, 0, 2, 3));

				posSrc = ctx0.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, pe_img.Shape[0], pe_img.Shape[1], pe_img.Shape[2], tokens.Shape[2]);
				posSrc = ctx0.Repeat(pe_img, posSrc);

				// flatten & permute
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
				posSrc = ctx0.Cont(ctx0.Permute(
					ctx0.View3d(
						posSrc,
						posSrc.Shape[0] * posSrc.Shape[1],
						posSrc.Shape[2],
						posSrc.Shape[3],
						posSrc.Stride[2],
						posSrc.Stride[3],
							0),
						1, 0, 2, 3));
			}

			SafeGGmlTensor queries = tokens;
			SafeGGmlTensor keys = src;
			{
				// Run the transformer
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
				for (int i = 0; i < model.decoderMask.transformer_layers.Length; ++i)
				{
					SamLayerDecTransformer tfmLayer = model.decoderMask.transformer_layers[i];

					// Self attention block
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
					bool skipFirstLayerPe = i == 0;
					if (skipFirstLayerPe)
					{
						queries = ctx0.SelfAttention(queries, tfmLayer.self_attn.q_w, tfmLayer.self_attn.q_b, tfmLayer.self_attn.k_w, tfmLayer.self_attn.k_b, tfmLayer.self_attn.v_w, tfmLayer.self_attn.v_b, tfmLayer.self_attn.out_w, tfmLayer.self_attn.out_b, model.hparams.DecoderHeadsCount);
					}
					else
					{

						SafeGGmlTensor q0 = ctx0.Add(queries, tokens);

						SafeGGmlTensor self_attn = ctx0.SelfAttention(q0, q0, queries, tfmLayer.self_attn.q_w, tfmLayer.self_attn.q_b, tfmLayer.self_attn.k_w, tfmLayer.self_attn.k_b, tfmLayer.self_attn.v_w, tfmLayer.self_attn.v_b, tfmLayer.self_attn.out_w, tfmLayer.self_attn.out_b, model.hparams.DecoderHeadsCount);
						queries = ctx0.Add(queries, self_attn);
					}

					queries = ctx0.LayerNorm(queries, tfmLayer.norm1_w, tfmLayer.norm1_b, hparams.EpsDecoderTransformer);

					// Cross attention block, tokens attending to image embedding
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
					SafeGGmlTensor q_1 = ctx0.Add(queries, tokens);
					SafeGGmlTensor k_1 = ctx0.Add(keys, posSrc);

					SafeGGmlTensor cross_attn_token_to_img = ctx0.SelfAttention(q_1, k_1, keys, tfmLayer.cross_attn_token_to_img.q_w, tfmLayer.cross_attn_token_to_img.q_b, tfmLayer.cross_attn_token_to_img.k_w, tfmLayer.cross_attn_token_to_img.k_b, tfmLayer.cross_attn_token_to_img.v_w, tfmLayer.cross_attn_token_to_img.v_b, tfmLayer.cross_attn_token_to_img.out_w, tfmLayer.cross_attn_token_to_img.out_b, model.hparams.DecoderHeadsCount);

					queries = ctx0.AddInplace(queries, cross_attn_token_to_img);

					queries = ctx0.LayerNorm(queries, tfmLayer.norm2_w, tfmLayer.norm2_b, hparams.EpsDecoderTransformer);


					// MLP block
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
					SafeGGmlTensor mlp_out = ctx0.Linear(queries, tfmLayer.mlp_lin1_w, tfmLayer.mlp_lin1_b);

					// RELU activation
					mlp_out = ctx0.ReluInplace(mlp_out);

					mlp_out = ctx0.Linear(mlp_out, tfmLayer.mlp_lin2_w, tfmLayer.mlp_lin2_b);

					queries = ctx0.AddInplace(queries, mlp_out);

					queries = ctx0.LayerNorm(queries, tfmLayer.norm3_w, tfmLayer.norm3_b, hparams.EpsDecoderTransformer);

					// Cross attention block, image embedding attending to tokens
					// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
					SafeGGmlTensor q_2 = ctx0.Add(queries, tokens);
					SafeGGmlTensor k_2 = ctx0.Add(keys, posSrc);

					SafeGGmlTensor cross_attn_img_to_token = ctx0.SelfAttention(k_2, q_2, queries, tfmLayer.cross_attn_img_to_token.q_w, tfmLayer.cross_attn_img_to_token.q_b, tfmLayer.cross_attn_img_to_token.k_w, tfmLayer.cross_attn_img_to_token.k_b, tfmLayer.cross_attn_img_to_token.v_w, tfmLayer.cross_attn_img_to_token.v_b, tfmLayer.cross_attn_img_to_token.out_w, tfmLayer.cross_attn_img_to_token.out_b, model.hparams.DecoderHeadsCount);
					keys = ctx0.AddInplace(keys, cross_attn_img_to_token);

					keys = ctx0.LayerNorm(keys, tfmLayer.norm4_w, tfmLayer.norm4_b, hparams.EpsDecoderTransformer);

				}

				// Apply the final attention layer from the points to the image
				// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
				SafeGGmlTensor q = ctx0.Add(queries, tokens);
				SafeGGmlTensor k = ctx0.Add(keys, posSrc);

				SafeGGmlTensor final_attn_token_to_img = ctx0.SelfAttention(q, k, keys, dec.transformer_final_attn_token_to_img.q_w, dec.transformer_final_attn_token_to_img.q_b, dec.transformer_final_attn_token_to_img.k_w, dec.transformer_final_attn_token_to_img.k_b, dec.transformer_final_attn_token_to_img.v_w, dec.transformer_final_attn_token_to_img.v_b, dec.transformer_final_attn_token_to_img.out_w, dec.transformer_final_attn_token_to_img.out_b, model.hparams.DecoderHeadsCount);

				queries = ctx0.AddInplace(queries, final_attn_token_to_img);
				queries = ctx0.LayerNorm(queries, dec.transformer_norm_final_w, dec.transformer_norm_final_b, hparams.EpsDecoderTransformer);
			}


			SafeGGmlTensor iou_pred = ctx0.View2d(queries, queries.Shape[0], queries.Shape[2], queries.Stride[2], 0);
			const int num_mask_tokens = 4; // num_multimask_outputs + 1
			SafeGGmlTensor mask_tokens_out = ctx0.View3d(queries, queries.Shape[0], num_mask_tokens, queries.Shape[2], queries.Stride[1], num_mask_tokens * queries.Stride[1], queries.Stride[1]);

			// Upscale mask embeddings and predict masks using the mask tokens
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
			keys = ctx0.Cont(ctx0.Transpose(keys));
			keys = ctx0.View4d(keys, srcNE[0], srcNE[1], srcNE[2], srcNE[3], (ulong)srcNE[0] * keys.Stride[0], keys.Stride[1], keys.Stride[2], 0);
			// ggml_build_forward_expand(graph, keys);
			SafeGGmlTensor upscaled_embedding;
			{
				// ConvTranspose2d
				keys = ctx0.ConvTranspose2dP0(dec.output_upscaling_0_w, keys, 2);
				keys = ctx0.AddInplace(keys, ctx0.Repeat(
											ctx0.Reshape3d(dec.output_upscaling_0_b, 1, 1, dec.output_upscaling_0_b.Shape[0]),
											 keys));

				keys = ctx0.LayerNorm2d(keys, n_img_embd, dec.output_upscaling_1_w, dec.output_upscaling_1_b, hparams.Eps);

				// GELU activation
				keys = ctx0.GeluInplace(keys);

				// ConvTranspose2d
				keys = ctx0.ConvTranspose2dP0(dec.output_upscaling_3_w, keys, 2);
				keys = ctx0.AddInplace(ctx0.Repeat(
										ctx0.Reshape3d(dec.output_upscaling_3_b, 1, 1, dec.output_upscaling_3_b.Shape[0]),
										keys), keys);
				// GELU activation
				keys = ctx0.GeluInplace(keys);

				upscaled_embedding = ctx0.Reshape3d(keys, keys.Shape[0] * keys.Shape[1], keys.Shape[2], keys.Shape[3]);
				upscaled_embedding = ctx0.Cont(ctx0.Transpose(upscaled_embedding)); // TODO: Shouldn't be needed
			}

			SafeGGmlTensor hyper_in = ctx0.NewTensor3d(Structs.GGmlType.GGML_TYPE_F32, n_img_embd / 2, num_mask_tokens, mask_tokens_out.Shape[2]);

			for (int i = 0; i < num_mask_tokens; ++i)
			{
				SamLayerDecOutputHypernetMlps mlp = dec.output_hypernet_mlps[i];

				SafeGGmlTensor input = ctx0.View2d(mask_tokens_out, mask_tokens_out.Shape[0], mask_tokens_out.Shape[2], mask_tokens_out.Stride[1], (ulong)i * mask_tokens_out.Stride[1]);
				SafeGGmlTensor output = SamDecodeMaskMlpRelu3(ctx0, input, mlp.w_0, mlp.b_0, mlp.w_1, mlp.b_1, mlp.w_2, mlp.b_2);
				gf.BuildForwardExpend(ctx0.Copy(output, ctx0.View2d(hyper_in, hyper_in.Shape[0], hyper_in.Shape[2], hyper_in.Stride[1], (ulong)i * hyper_in.Stride[1])));
			}

			SafeGGmlTensor masks = ctx0.MulMat(hyper_in, upscaled_embedding);
			masks = ctx0.Cont(ctx0.Transpose(masks)); // TODO: Shouldn't be needed
			masks = ctx0.Reshape4d(masks, keys.Shape[0], keys.Shape[1], masks.Shape[1], keys.Shape[3]);

			// Generate mask quality predictions
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
			iou_pred = SamDecodeMaskMlpRelu3(ctx0, iou_pred, dec.iou_prediction_head_0_w, dec.iou_prediction_head_0_b, dec.iou_prediction_head_1_w, dec.iou_prediction_head_1_b, dec.iou_prediction_head_2_w, dec.iou_prediction_head_2_b);

			// Select the correct mask or masks for output
			// ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
			iou_pred = state.context.Copy(ctx0.View1d(iou_pred, iou_pred.Shape[0] - 1, iou_pred.Stride[0]), state.iouPredictions);
			masks = ctx0.View4d(masks, masks.Shape[0], masks.Shape[1], masks.Shape[2] - 1, masks.Shape[3], masks.Stride[1], masks.Stride[2], masks.Stride[3], masks.Stride[2] /* offset*/);
			masks = state.context.Copy(masks, state.lowResMasks);
			gf.BuildForwardExpend(masks);
			gf.BuildForwardExpend(iou_pred);
			DisconnectNodeFromGraph(state.lowResMasks);
			DisconnectNodeFromGraph(state.iouPredictions);

			return true;
		}

		static SafeGGmlGraph SamBuildFastGraph(SamModel model, SamState state, int nx, int ny, PointF point)
		{
			ulong size = Common.TensorOverheadLength * GGML_DEFAULT_GRAPH_SIZE + Common.GraphOverheadLength;
			SafeGGmlContext ctx0 = new SafeGGmlContext(IntPtr.Zero, 1024 * 1024 * 1024, true);
			SafeGGmlGraph gf = ctx0.NewGraph();

			PromptEncoderResult enc_res = SamEncodePrompt(model, ctx0, gf, state);
			if (enc_res.embdPromptSparse == null || enc_res.embdPromptDense == null)
			{
				throw new Exception($"failed to encode prompt ({point.X}, {point.Y})");
			}

			SafeGGmlTensor pe_img_dense = SamFillDensePe(model, ctx0, gf, state);
			if (pe_img_dense == null)
			{
				throw new Exception("failed to get dense positional encoding");
			}

			if (!SamDecodeMask(model, enc_res, pe_img_dense, ctx0, gf, state))
			{
				throw new Exception("failed to decode mask");
			}

			//Native.ggml_free(ctx0);
			gf.GraphAllocate(state.allocr);

			// from SamEncodePrompt
			{
				// transform points
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/automatic_mask_generator.py#L276
				{
					int nmax = Math.Max(nx, ny);

					float scale = model.hparams.ImgSize / (float)nmax;

					int nx_new = (int)(nx * scale + 0.5f);
					int ny_new = (int)(ny * scale + 0.5f);

					point.X = point.X * ((float)(nx_new) / nx) + 0.5f;
					point.Y = point.Y * ((float)(ny_new) / ny) + 0.5f;
				}


				SafeGGmlTensor inp = gf.GetTensor("prompt_input");
				// set the input by converting the [0, 1] coordinates to [-1, 1]

				float[] data = new float[4];
				data[0] = 2.0f * (point.X / model.hparams.ImgSize) - 1.0f;
				data[1] = 2.0f * (point.Y / model.hparams.ImgSize) - 1.0f;

				// padding
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L81-L85
				data[2] = 2.0f * (0.0f) - 1.0f;
				data[3] = 2.0f * (0.0f) - 1.0f;

				inp.SetData(data);
			}

			// from SamFillDensePe
			{
				SafeGGmlTensor xy_embed_stacked = gf.GetTensor("xy_embed_stacked");
				int n_img_embd = model.hparams.ImgEmbdCount;
				float n_img_embd_inv = 1.0f / n_img_embd;
				float[] data = new float[xy_embed_stacked.ElementsCount];
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

				xy_embed_stacked.SetData(data);
			}

			return gf;
		}

		static SafeGGmlGraph SamEncodeImage(SamModel model, SamState state, SamImageF32 img)
		{
			SamHparams hparams = model.hparams;
			SamEncoderImage enc = model.encoderImg;

			int n_enc_state = hparams.EncoderState;
			int n_enc_layer = hparams.EncoderLayer;
			int n_enc_head = hparams.EncoderHead;
			int n_enc_head_dim = hparams.EncHeadDim;
			int n_enc_out_chans = hparams.EncoderOutChans;
			int n_img_size = hparams.ImgSize;
			int n_window_size = hparams.WindowSize;

			ulong mem_size = 512u * 1024 * 1024;
			SafeGGmlContext ctx0 = new SafeGGmlContext(IntPtr.Zero, mem_size, true);
			SafeGGmlGraph gf = ctx0.NewGraph();

			SafeGGmlTensor inp = ctx0.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, n_img_size, n_img_size, 3, 1);
			inp.Name = "inp";
			inp.SetInput();

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
			SafeGGmlTensor cur = ctx0.Conv2dSkP0(enc.proj_w, inp);
			cur = ctx0.AddInplace(
					cur,
					ctx0.Repeat(enc.proj_b, cur));

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
			// keep in F32
			cur = ctx0.Cont(
					ctx0.Permute(cur, 1, 2, 0, 3));

			// convert to F16
			//cur = ggml_cpy(ctx0,
			//        Permute(ctx0, cur, 1, 2, 0, 3),
			//        ggml_new_tensor_3d(ctx0, GGML_TYPE_F16, EncoderState, ImgEmbdCount, ImgEmbdCount));

			// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
			cur = ctx0.AddInplace(cur, enc.pe);

			SafeGGmlTensor inpL = cur;

			for (int il = 0; il < n_enc_layer; ++il)
			{
				SamLayerEnc layer = enc.layers[il];

				// norm
				// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
				{
					cur = ctx0.Normal(inpL, hparams.Eps);

					// cur = ln_0_w*cur + ln_0_b
					cur = ctx0.Mul(cur, layer.norm1_w);
					cur = ctx0.AddInplace(cur, layer.norm1_b);
				}

				long w0 = cur.Shape[1];
				long h0 = cur.Shape[2];

				if (hparams.IsGlobalAttn(il) == false)
				{
					// local attention layer - apply window partition
					// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
					cur = ctx0.WinPart(cur, n_window_size);
				}
				long W = cur.Shape[1];
				long H = cur.Shape[2];

				// self-attention
				{
					cur = ctx0.MulMat(layer.qkv_w, cur);
					cur = ctx0.AddInplace(cur, layer.qkv_b);

					// split qkv into separate tensors
					// ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
					long B = cur.Shape[3];

					cur = ctx0.Reshape4d(cur, n_enc_state, 3, W * H, B);
					cur = ctx0.Cont(ctx0.Permute(cur, 0, 3, 1, 2));


					SafeGGmlTensor Q;
					SafeGGmlTensor K;
					SafeGGmlTensor V;

					Q = ctx0.View3d(cur, n_enc_state, W * H, B, cur.Stride[1], cur.Stride[2], 0 * cur.Stride[3]);
					Q = ctx0.Reshape4d(Q, n_enc_head_dim, n_enc_head, W * H, B);
					Q = ctx0.Cont(ctx0.Permute(Q, 0, 2, 1, 3));
					Q = ctx0.Reshape3d(Q, n_enc_head_dim, W * H, B * n_enc_head);

					K = ctx0.View3d(cur, n_enc_state, W * H, B, cur.Stride[1], cur.Stride[2], 1 * cur.Stride[3]);
					K = ctx0.Reshape4d(K, n_enc_head_dim, n_enc_head, W * H, B);
					K = ctx0.Cont(ctx0.Permute(K, 0, 2, 1, 3));
					K = ctx0.Reshape3d(K, n_enc_head_dim, W * H, B * n_enc_head);
					V = ctx0.View3d(cur, n_enc_state, W * H, B, cur.Stride[1], cur.Stride[2], 2 * cur.Stride[3]);
					V = ctx0.Reshape4d(V, n_enc_head_dim, n_enc_head, W * H, B);
					V = ctx0.Cont(ctx0.Permute(V, 1, 2, 0, 3)); // transposed
					V = ctx0.Reshape3d(V, W * H, n_enc_head_dim, B * n_enc_head);

					SafeGGmlTensor KQ = ctx0.MulMat(K, Q);

					SafeGGmlTensor KQ_scaled = ctx0.Scale(KQ, 1.0f / (float)Math.Sqrt(n_enc_head_dim));

					SafeGGmlTensor rw = ctx0.GetRelPos(layer.rel_pos_w, (int)W, (int)W);
					SafeGGmlTensor rh = ctx0.GetRelPos(layer.rel_pos_h, (int)H, (int)H);

					SafeGGmlTensor q_r = ctx0.Reshape4d(Q, n_enc_head_dim, W, H, B * n_enc_head);

					SafeGGmlTensor rel_w = ctx0.Cont(ctx0.Permute(
							  ctx0.MulMat(
								  rw,
								 ctx0.Cont(ctx0.Permute(q_r, 0, 2, 1, 3))),
							  0, 2, 1, 3));
					SafeGGmlTensor rel_h = ctx0.MulMat(rh, q_r);

					SafeGGmlTensor attn = ctx0.AddRelPosInplace(KQ_scaled, rel_w, rel_h);

					SafeGGmlTensor KQ_soft_max = ctx0.SoftmaxInplace(attn);

					SafeGGmlTensor KQV = ctx0.MulMat(V, KQ_soft_max);

					cur =
						ctx0.Reshape4d(
								ctx0.Cont(
									ctx0.Permute(
										ctx0.Reshape4d(KQV, n_enc_head_dim, W * H, n_enc_head, B),
					0, 2, 1, 3)),
					n_enc_state, W, H, B);

					cur = ctx0.MulMat(layer.proj_w, cur);
					cur = ctx0.AddInplace(cur, layer.proj_b);
				}

				if (hparams.IsGlobalAttn(il) == false)
				{
					// local attention layer - reverse window partition
					cur = ctx0.WinUnpart(cur, (int)w0, (int)h0, n_window_size);
				}

				cur = ctx0.AddInplace(cur, inpL);

				SafeGGmlTensor inpFF = cur;

				// feed-forward network
				{
					// norm
					{
						cur = ctx0.Normal(inpFF, hparams.Eps);

						// cur = mlp_ln_w*cur + mlp_ln_b
						cur = ctx0.Mul(cur, layer.norm2_w);
						cur = ctx0.AddInplace(cur, layer.norm2_b);
					}

					// fully connected
					cur = ctx0.MulMat(layer.mlp_lin1_w, cur);
					cur = ctx0.AddInplace(cur, layer.mlp_lin1_b);

					// GELU activation
					cur = ctx0.Gelu(cur);

					// projection
					cur = ctx0.MulMat(layer.mlp_lin2_w, cur);
					cur = ctx0.AddInplace(cur, layer.mlp_lin2_b);
				}

				inpL = ctx0.Add(cur, inpFF);
			}

			cur = ctx0.Cont(ctx0.Permute(inpL, 2, 0, 1, 3));
			cur = ctx0.Conv2dSkP0(enc.neck_conv_0, cur);
			cur = ctx0.LayerNorm2d(cur, n_enc_out_chans, enc.neck_norm_0_w, enc.neck_norm_0_b, hparams.Eps);
			cur = ctx0.Conv2dS1Ph(enc.neck_conv_1, cur);
			cur = ctx0.LayerNorm2d(cur, n_enc_out_chans, enc.neck_norm_1_w, enc.neck_norm_1_b, hparams.Eps);

			cur = ctx0.Copy(cur, state.embdImg);

			gf.BuildForwardExpend(cur);
			DisconnectNodeFromGraph(state.embdImg);

			//ggml_graph_print(&graph);

			//ctx0.Free();
			gf.GraphAllocate(state.allocr);
			{
				SafeGGmlTensor inpt = gf.GetTensor("inp");
				float[] data = new float[inpt.ElementsCount];
				int nx = img.Width;
				int ny = img.Height;
				int n = nx * ny;

				if (!(nx == n_img_size && ny == n_img_size))
				{
					throw new Exception("Width == ImgSize && Height == ImgSize wrong");
				}

				for (int k = 0; k < 3; k++)
				{
					for (int y = 0; y < ny; y++)
					{
						for (int x = 0; x < nx; x++)
						{
							data[k * n + y * nx + x] = img.Data[3 * (y * nx + x) + k];
						}
					}
				}
				inpt.SetData(data);
			}
			return gf;
		}

		static bool WriteMasks(SamHparams hparams, int nx, int ny, SamState state, string fname)
		{
			if (state.lowResMasks.Shape[2] == 0) return true;
			if (state.lowResMasks.Shape[2] != state.iouPredictions.Shape[0])
			{
				throw new ArgumentException($"Error: number of masks ({state.lowResMasks.Shape[2]}) does not match number of iou predictions ({state.iouPredictions.Shape[0]})");
			}

			int n_img_size = hparams.ImgSize;
			float mask_threshold = hparams.MaskThreshold;
			float iou_threshold = hparams.IouThreshold;
			float stability_score_threshold = hparams.StabilityScoreThreshold;
			float intersection_threshold = mask_threshold + hparams.StabilityScoreOffset;
			float union_threshold = mask_threshold - hparams.StabilityScoreOffset;

			int ne0 = (int)state.lowResMasks.Shape[0];
			int ne1 = (int)state.lowResMasks.Shape[1];
			int ne2 = (int)state.lowResMasks.Shape[2];

			// Remove padding and upscale masks to the original image size.
			// ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

			float preprocess_scale = (float)Math.Max(nx, ny) / (float)(n_img_size);
			int cropped_nx = (int)(nx / preprocess_scale + 0.5f);
			int cropped_ny = (int)(ny / preprocess_scale + 0.5f);

			float scale_x_1 = (float)ne0 / (float)n_img_size;
			float scale_y_1 = (float)ne1 / (float)n_img_size;

			float scale_x_2 = (float)(cropped_nx) / (float)(nx);
			float scale_y_2 = (float)(cropped_ny) / (float)(ny);

			float[] iou_data = DataConverter.ConvertToFloats(state.iouPredictions.GetData());

			for (int i = 0; i < ne2; ++i)
			{
				if (iou_threshold > 0.0f && iou_data[i] < iou_threshold)
				{
					Console.WriteLine($"Skipping mask {i} with iou {iou_data[i]} below threshold {iou_threshold}");
					continue; // Filtering masks with iou below the threshold
				}

				float[] mask_data = new float[n_img_size * n_img_size];
				{
					//float[] Data = new float[state.lowResMasks.ElementsCount];
					//Marshal.Copy(state.lowResMasks.Data + i * ne0 * ne1, Data, 0, (int)state.lowResMasks.ElementsCount);
					//float* Data = (float*)state.lowResMasks.Data + i * ne0 * ne1;
					float[] data = new float[ne0 * ne1];
					Marshal.Copy(state.lowResMasks.Data + i * ne0 * ne1 * 4, data, 0, ne0 * ne1);
					for (int iy = 0; iy < n_img_size; ++iy)
					{
						for (int ix = 0; ix < n_img_size; ++ix)
						{
							float sx = (float)Math.Max(scale_x_1 * (ix + 0.5f) - 0.5f, 0.0f);
							float sy = (float)Math.Max(scale_y_1 * (iy + 0.5f) - 0.5f, 0.0f);

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
				SamImageU8 res;
				int min_iy = ny;
				int max_iy = 0;
				int min_ix = nx;
				int max_ix = 0;
				{
					float[] data = mask_data;

					res.Width = nx;
					res.Height = ny;
					res.Data = new byte[nx * ny];

					for (int iy = 0; iy < ny; ++iy)
					{
						for (int ix = 0; ix < nx; ++ix)
						{
							float sx = (float)Math.Max(scale_x_2 * (ix + 0.5f) - 0.5f, 0.0f);
							float sy = (float)Math.Max(scale_y_2 * (iy + 0.5f) - 0.5f, 0.0f);

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

								res.Data[iy * nx + ix] = 255;
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

				Bitmap bitmap = new Bitmap(res.Width, res.Height, PixelFormat.Format8bppIndexed);
				BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, res.Width, res.Height), ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
				Marshal.Copy(res.Data, 0, bitmapData.Scan0, res.Data.Length);
				bitmap.UnlockBits(bitmapData);
				bitmap.Save(filename, ImageFormat.Png);
			}


			return true;
		}

	}
}

