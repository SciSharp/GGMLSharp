using GGMLSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;

namespace Yolov3Tiny
{
	internal class Program
	{
		static void Main(string[] args)
		{
			string modelPath = @".\Assets\yolov3-tiny.gguf";
			string inputImgPath = @".\Assets\test.jpg";
			string outputImgPath = @".\output.jpg";
			string labelPath = @".\Assets\coco.names";

			int modelWidth = 416;
			int modelHeight = 416;
			float classThresh = 0.25f;
			float nmsThresh = 0.5f;

			string[] labels = File.ReadAllLines(labelPath);
			int classes = labels.Length;

			SafeGGmlContext ctx0 = new SafeGGmlContext(IntPtr.Zero, 512 * 1024 * 1024, false);

			YoloModel model = LoadModel(modelPath);

			Bitmap inputImg = new Bitmap(inputImgPath);
			Bitmap resizedImg = ResizeImage(inputImg, modelWidth, modelHeight, out float ratio);

			BitmapData bitmapData = resizedImg.LockBits(new Rectangle(0, 0, resizedImg.Width, resizedImg.Height), ImageLockMode.ReadOnly, resizedImg.PixelFormat);
			byte[] data = new byte[3 * modelWidth * modelHeight];
			Marshal.Copy(bitmapData.Scan0, data, 0, data.Length);
			SafeGGmlTensor input = ctx0.NewTensor4d(Structs.GGmlType.GGML_TYPE_F32, modelWidth, modelHeight, 3, 1);
			input.Name = "input";
			for (int w = 0; w < modelWidth; w++)
			{
				for (int h = 0; h < modelHeight; h++)
				{
					input.SetData(data[bitmapData.Stride * h + w * 3 + 2] / 255.0f, w, h, 0, 0);
					input.SetData(data[bitmapData.Stride * h + w * 3 + 1] / 255.0f, w, h, 1, 0);
					input.SetData(data[bitmapData.Stride * h + w * 3 + 0] / 255.0f, w, h, 2, 0);
				}
			}
			resizedImg.UnlockBits(bitmapData);

			SafeGGmlTensor result = ApplyConv2d(ctx0, input, model.Conv2dLayers[0]);
			PrintShape(0, result);
			result = ctx0.Pool2d(result);
			PrintShape(1, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[1]);
			PrintShape(2, result);
			result = ctx0.Pool2d(result);
			PrintShape(3, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[2]);
			PrintShape(4, result);
			result = ctx0.Pool2d(result);
			PrintShape(5, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[3]);
			PrintShape(6, result);
			result = ctx0.Pool2d(result);
			PrintShape(7, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[4]);

			SafeGGmlTensor layer8 = result;
			PrintShape(8, result);
			result = ctx0.Pool2d(result);
			PrintShape(9, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[5]);
			PrintShape(10, result);
			result = ctx0.Pool2d(result, Structs.GGmlOpPool.GGML_OP_POOL_MAX, 2, 2, 1, 1, 0.5f, 0.5f);
			PrintShape(11, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[6]);
			PrintShape(12, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[7]);
			SafeGGmlTensor layer13 = result;
			PrintShape(13, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[8]);
			PrintShape(14, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[9]);
			SafeGGmlTensor layer15 = result;
			layer15.Name = "layer15";
			PrintShape(15, result);
			result = ApplyConv2d(ctx0, layer13, model.Conv2dLayers[10]);
			PrintShape(18, result);
			result = ctx0.Upscale(result, 2);
			PrintShape(19, result);
			result = ctx0.Concat(result, layer8, 2);
			PrintShape(20, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[11]);
			PrintShape(21, result);
			result = ApplyConv2d(ctx0, result, model.Conv2dLayers[12]);
			SafeGGmlTensor layer_22 = result;
			layer_22.Name = "layer22";
			PrintShape(22, result);

			SafeGGmlGraph gf = ctx0.NewGraph();
			gf.BuildForwardExpend(layer15);
			gf.BuildForwardExpend(layer_22);
			gf.ComputeWithGGmlContext(ctx0, 1);

			YoloLayer yolo16 = new YoloLayer(classes, new int[] { 3, 4, 5 }, new float[] { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 }, layer15);
			ApplyYolo(yolo16);

			List<Detection> detections = new List<Detection>();
			detections.AddRange(GetYoloDetections(yolo16, resizedImg.Width, resizedImg.Height, model.Width, model.Height, classThresh));

			YoloLayer yolo23 = new YoloLayer(classes, new int[] { 0, 1, 2 }, new float[] { 10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319 }, layer_22);
			ApplyYolo(yolo23);

			detections.AddRange(GetYoloDetections(yolo23, resizedImg.Width, resizedImg.Height, model.Width, model.Height, classThresh));

			detections = DoNmsSort(detections, classes, nmsThresh);


			Console.WriteLine();

			using (Graphics g = Graphics.FromImage(inputImg))
			{
				int d = Math.Abs(inputImg.Width - inputImg.Height) / 2;
				bool wIsLonger = inputImg.Width.CompareTo(inputImg.Height) > 0;

				foreach (Detection detection in detections)
				{

					int w = (int)(detection.BBox.W * resizedImg.Width / ratio);
					int h = (int)(detection.BBox.H * resizedImg.Height / ratio);
					int x = (int)(detection.BBox.X * resizedImg.Width / ratio - w / 2) - (wIsLonger ? 0 : d);
					int y = (int)(detection.BBox.Y * resizedImg.Height / ratio - h / 2) - (wIsLonger ? d : 0);
					Rectangle rect = new Rectangle(x, y, w, h);
					g.DrawRectangle(Pens.Red, rect);

					int index = detection.Prob.ToList().IndexOf(detection.Prob.Max());
					string str = labels[index] + "  " + (detection.Objectness * 100).ToString("f2") + "%";
					SolidBrush redBrush = new SolidBrush(Color.Red);
					Font font = new Font("Arial", 16);
					g.DrawString(str, font, redBrush, x, y);
					g.Save();
					Console.WriteLine(string.Format("Detect: " + str));
				}
			}

			Console.WriteLine();
			inputImg.Save(outputImgPath);
			Console.WriteLine("Done.");
			Console.ReadLine();
		}

		static List<Detection> GetYoloDetections(YoloLayer layer, int im_w, int im_h, int netw, int neth, float thresh)
		{
			List<Detection> detections = new List<Detection>();
			int w = (int)layer.Predictions.Shape[0];
			int h = (int)layer.Predictions.Shape[1];

			float[] predictions = layer.Predictions.GetDataInFloats();
			for (int i = 0; i < w * h; i++)
			{
				for (int n = 0; n < layer.Mask.Length; n++)
				{
					int obj_index = layer.GetEntryIndex(n * w * h + i, 4);
					float objectness = predictions[obj_index];
					if (objectness <= thresh)
					{
						continue;
					}
					Detection det = new Detection();
					int box_index = layer.GetEntryIndex(n * w * h + i, 0);
					int row = i / w;
					int col = i % w;
					det.BBox = GetYoloBox(layer, layer.Mask[n], box_index, col, row, w, h, netw, neth, w * h);
					det.Objectness = objectness;
					det.Prob = new float[layer.Classes];
					for (int j = 0; j < layer.Classes; j++)
					{
						int class_index = layer.GetEntryIndex(n * w * h + i, 4 + 1 + j);
						float prob = objectness * predictions[class_index];
						det.Prob[j] = prob;
					}
					if (det.Prob.ToList().Max() > thresh)
					{
						detections.Add(det);
					}
				}
			}
			return detections;
		}

		static void PrintShape(int layer, SafeGGmlTensor t)
		{
			Console.WriteLine(string.Format("Layer {0} output shape:  {1} x {2} x {3} x {4}", layer, (int)t.Shape[0], (int)t.Shape[1], (int)t.Shape[2], (int)t.Shape[3]));
		}

		public static Bitmap ResizeImage(Bitmap image, int targetWidth, int targetHeight, out float ratio)
		{
			PixelFormat format = image.PixelFormat;
			Bitmap output = new Bitmap(targetWidth, targetHeight, format);
			int w = image.Width;
			int h = image.Height;
			float xRatio = targetWidth / (float)w;
			float yRatio = targetHeight / (float)h;
			ratio = Math.Min(xRatio, yRatio);
			int width = (int)(w * ratio);
			int height = (int)(h * ratio);
			int x = targetWidth / 2 - width / 2;
			int y = targetHeight / 2 - height / 2;
			Rectangle roi = new Rectangle(x, y, width, height);
			using (Graphics graphics = Graphics.FromImage(output))
			{
				graphics.Clear(Color.Black);
				graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
				graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
				graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
				graphics.DrawImage(image, roi);
			}
			return output;
		}

		public class Conv2dLayer
		{
			public SafeGGmlTensor Weights;
			public SafeGGmlTensor Biases;
			public SafeGGmlTensor Scales;
			public SafeGGmlTensor RollingMean;
			public SafeGGmlTensor RollingVariance;
			public int Padding = 1;
			public bool NatchNormalize = true;
			public bool Activate = true; // true for leaky relu, false for linear
		};

		public class YoloModel
		{
			public int Width = 416;
			public int Height = 416;
			public Conv2dLayer[] Conv2dLayers = new Conv2dLayer[13];
			public SafeGGmlContext context = new SafeGGmlContext();
		};

		public class YoloLayer
		{
			public int Classes = 80;
			public int[] Mask;
			public float[] Anchors;
			public SafeGGmlTensor Predictions;

			public YoloLayer(int classes, int[] mask, float[] anchors, SafeGGmlTensor predictions)
			{
				this.Classes = classes;
				this.Mask = mask;
				this.Anchors = anchors;
				this.Predictions = predictions;
			}

			public int GetEntryIndex(int location, int entry)
			{
				int w = (int)Predictions.Shape[0];
				int h = (int)Predictions.Shape[1];
				int n = location / (w * h);
				int loc = location % (w * h);
				return n * w * h * (4 + Classes + 1) + entry * w * h + loc;
			}
		};
		public class Box
		{
			public float X, Y, W, H;
		};

		public class Detection
		{
			public Box BBox;
			public float[] Prob;
			public float Objectness;
			public int ForcastIndex
			{
				get
				{
					return Prob.ToList().IndexOf(Prob.Max());
				}
			}
		};

		public static YoloModel LoadModel(string fname)
		{
			YoloModel model = new YoloModel();
			SafeGGufContext ctx = SafeGGufContext.InitFromFile(fname, model.context, false);
			if (ctx.IsInvalid)
			{
				throw new ArgumentNullException("GGuf context is null.");
			}
			for (int i = 0; i < model.Conv2dLayers.Length; i++)
			{
				model.Conv2dLayers[i] = new Conv2dLayer();
			}
			model.Conv2dLayers[7].Padding = 0;
			model.Conv2dLayers[9].Padding = 0;
			model.Conv2dLayers[9].NatchNormalize = false;
			model.Conv2dLayers[9].Activate = false;
			model.Conv2dLayers[10].Padding = 0;
			model.Conv2dLayers[12].Padding = 0;
			model.Conv2dLayers[12].NatchNormalize = false;
			model.Conv2dLayers[12].Activate = false;
			for (int i = 0; i < model.Conv2dLayers.Length; i++)
			{
				string name = "l" + i + "_weights";
				model.Conv2dLayers[i].Weights = model.context.GetTensor(name);
				name = "l" + i + "_biases";
				model.Conv2dLayers[i].Biases = model.context.GetTensor(name);
				if (model.Conv2dLayers[i].NatchNormalize)
				{
					name = "l" + i + "_scales";
					model.Conv2dLayers[i].Scales = model.context.GetTensor(name);
					name = "l" + i + "_rolling_mean";
					model.Conv2dLayers[i].RollingMean = model.context.GetTensor(name);
					name = "l" + i + "_rolling_variance";
					model.Conv2dLayers[i].RollingVariance = model.context.GetTensor(name);
				}
			}
			return model;
		}

		public static SafeGGmlTensor ApplyConv2d(SafeGGmlContext ctx, SafeGGmlTensor input, Conv2dLayer layer)
		{
			SafeGGmlTensor result = ctx.Conv2d(layer.Weights, input, 1, 1, layer.Padding, layer.Padding, 1, 1);
			if (layer.NatchNormalize)
			{
				result = ctx.Sub(result, ctx.Repeat(layer.RollingMean, result));
				result = ctx.Div(result, ctx.Sqrt(ctx.Repeat(layer.RollingVariance, result)));
				result = ctx.Mul(result, ctx.Repeat(layer.Scales, result));
			}
			result = ctx.Add(result, ctx.Repeat(layer.Biases, result));
			if (layer.Activate)
			{
				result = ctx.LeakyRelu(result, 0.1f, true);
			}
			return result;

		}

		public static void ApplyYolo(YoloLayer layer)
		{
			int w = (int)layer.Predictions.Shape[0];
			int h = (int)layer.Predictions.Shape[1];
			float[] data = layer.Predictions.GetDataInFloats();

			for (int n = 0; n < layer.Mask.Length; n++)
			{
				int index = layer.GetEntryIndex(n * w * h, 0);

				for (int i = 0; i < 2 * w * h; i++)
				{
					data[index + i] = (float)(1.0f / (1.0f + Math.Exp(-data[index + i])));
				}
				index = layer.GetEntryIndex(n * w * h, 4);
				for (int i = 0; i < (1 + layer.Classes) * w * h; i++)
				{
					data[index + i] = (float)(1.0f / (1.0f + Math.Exp(-data[index + i])));
				}
			}
			layer.Predictions.SetData(data);
		}

		public static Box GetYoloBox(YoloLayer layer, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
		{

			float[] predictions = layer.Predictions.GetDataInFloats();
			Box b = new Box();
			b.X = (i + predictions[index + 0 * stride]) / lw;
			b.Y = (j + predictions[index + 1 * stride]) / lh;
			b.W = (float)Math.Exp(predictions[index + 2 * stride]) * layer.Anchors[2 * n] / w;
			b.H = (float)Math.Exp(predictions[index + 3 * stride]) * layer.Anchors[2 * n + 1] / h;
			return b;
		}


		public static List<Detection> DoNmsSort(List<Detection> dets, int classes, float thresh)
		{
			dets.Sort((a, b) => { return b.Objectness.CompareTo(a.Objectness); });

			bool[] selectedList = new bool[dets.Count];
			for (int i = 0; i < selectedList.Length; i++)
			{
				selectedList[i] = true;
			}

			for (int i = 0; i < dets.Count; i++)
			{
				for (int j = i + 1; j < dets.Count; j++)
				{
					if (selectedList[j])
					{
						if (BoxIou(dets[i], dets[j]) > thresh)
						{
							selectedList[j] = false;
						}
					}
				}
			}
			List<Detection> re = new List<Detection>();
			for (int i = 0; i < selectedList.Length; i++)
			{
				if (selectedList[i])
				{
					re.Add(dets[i]);
				}
			}
			return re;
		}

		private static float BoxIou(Detection detA, Detection detB)
		{
			if (detA.ForcastIndex != detB.ForcastIndex)
			{
				return 0;
			}

			Rectangle a = new Rectangle((int)(detA.BBox.X * 100), (int)(detA.BBox.Y * 100), (int)(detA.BBox.W * 100), (int)(detA.BBox.H * 100));
			Rectangle b = new Rectangle((int)(detB.BBox.X * 100), (int)(detB.BBox.Y * 100), (int)(detB.BBox.W * 100), (int)(detB.BBox.H * 100));

			float areaA = a.Width * a.Height;
			float areaB = b.Width * b.Height;
			float area = Math.Min(areaA, areaB);

			a.Intersect(b);
			float ins = a.Width * a.Height / area;
			if (float.IsNaN(ins))
			{
				return 0;
			}
			else
			{
				return ins;
			}
		}

	}




}

