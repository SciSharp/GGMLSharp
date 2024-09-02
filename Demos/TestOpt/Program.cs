using GGMLSharp;
using System;
using static GGMLSharp.Structs;

namespace TestOpt
{
	internal class Program
	{
		static void Main(string[] args)
		{
			long[] ne1 = { 4, 128, 1, 1 };
			long[] ne2 = { 4, 256, 1, 1 };
			long[] ne3 = { 128, 256, 1, 1 };

			SafeGGmlContext ctx = new SafeGGmlContext();

			SafeGGmlTensor a = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne1);
			SafeGGmlTensor b = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne2);
			SafeGGmlTensor c = new SafeGGmlTensor(ctx, Structs.GGmlType.GGML_TYPE_F32, ne3);
			a.SetRandomTensorInFloat(-1.0f, 1.0f);
			b.SetRandomTensorInFloat(-1.0f, 1.0f);
			c.SetRandomTensorInFloat(-1.0f, 1.0f);

			ctx.SetParam(a);
			ctx.SetParam(b);

			SafeGGmlTensor ab = ctx.MulMat(a, b);
			SafeGGmlTensor d = ctx.Sub(c, ab);
			SafeGGmlTensor e = ctx.Sum(ctx.Sqr(d));

			SafeGGmlGraph ge = ctx.CustomNewGraph(GGML_DEFAULT_GRAPH_SIZE, true);
			ge.BuildForwardExpend(e);
			ge.Reset();

			ge.ComputeWithGGmlContext(ctx, /*Threads*/ 1);

			float fe = e.GetFloat();
			Console.WriteLine("e = " + fe);

			OptimizerParameters optParams = SafeGGmlContext.GetDefaultOptimizerParams(OptimizerType.ADAM);

			ctx.Optimizer(optParams, e);

			ge.Reset();

			ge.ComputeWithGGmlContext(ctx, /*Threads*/ 1);

			float fe_opt = e.GetFloat();
			Console.WriteLine("original  e = " + fe);
			Console.WriteLine("optimized e = " + fe_opt);

			bool success = fe_opt < fe;
			ctx.Free();
			Console.WriteLine("success:" + success);
			Console.ReadKey();

		}

	}
}
