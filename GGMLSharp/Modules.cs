using System;

namespace GGMLSharp
{
	public class Modules
	{
		public class Linear
		{
			public SafeGGmlTensor Weight { get; set; }
			public SafeGGmlTensor Bias { get; set; }

			public Linear()
			{

			}

			public Linear(SafeGGmlTensor weight, SafeGGmlTensor bias)
			{
				this.Weight = weight;
				this.Bias = bias;
			}

			public Linear(SafeGGmlContext context, Structs.GGmlType type, long in_features, long out_features)
			{
				Weight = context.NewTensor2d(type, in_features, out_features);
				Bias = context.NewTensor1d(type, out_features);
			}

			public SafeGGmlTensor Forward(SafeGGmlContext context, SafeGGmlTensor input)
			{
				if (Weight == null || Weight.IsInvalid)
				{
					throw new ArgumentNullException(nameof(Weight));
				}

				if (Bias.IsInvalid)
				{
					return context.MulMat(Weight, input);
				}
				else
				{
					return context.Linear(input, Weight, Bias);
				}
			}
		}

		public class LayerNorm
		{
			public SafeGGmlTensor Weight { get; set; }
			public SafeGGmlTensor Bias { get; set; }
			public float Eps { get; set; } = 1e-5f;

			public LayerNorm(SafeGGmlTensor weight, SafeGGmlTensor bias, float eps = 1e-5f)
			{
				Weight = weight;
				Bias = bias;
				Eps = eps;
			}

			public LayerNorm()
			{

			}

			public SafeGGmlTensor Forward(SafeGGmlContext context, SafeGGmlTensor input)
			{
				if (Weight == null || Weight.IsInvalid)
				{
					throw new ArgumentNullException(nameof(Weight));
				}

				SafeGGmlTensor re = context.Norm(input, Eps);
				re = context.Mul(re, Weight);
				if (Bias.IsInvalid)
				{
					re = context.Add(re, Bias);
				}
				return re;
			}
		}

		public class SelfAttention
		{
			public Linear qLinear { get; set; }
			public Linear kLinear { get; set; }
			public Linear vLinear { get; set; }
			public Linear outLinear { get; set; }

			public SelfAttention(Linear qLinear, Linear kLinear, Linear vLinear, Linear outLinear)
			{
				this.qLinear = qLinear;
				this.kLinear = kLinear;
				this.vLinear = vLinear;
				this.outLinear = outLinear;
			}

			public SelfAttention(SafeGGmlTensor qWeight, SafeGGmlTensor qBias, SafeGGmlTensor kWeight, SafeGGmlTensor kBias, SafeGGmlTensor vWeight, SafeGGmlTensor vBias, SafeGGmlTensor outWeight, SafeGGmlTensor outBias)
			{
				this.qLinear = new Linear(qWeight, qBias);
				this.kLinear = new Linear(kWeight, kBias);
				this.vLinear = new Linear(vWeight, vBias);
				this.outLinear = new Linear(outWeight, outBias);
			}

			public SafeGGmlTensor Forward(SafeGGmlContext context, SafeGGmlTensor queries, SafeGGmlTensor keys, SafeGGmlTensor values, long headers)
			{
				SafeGGmlTensor qCur = qLinear.Forward(context, queries);
				SafeGGmlTensor kCur = kLinear.Forward(context, keys);
				SafeGGmlTensor vCur = vLinear.Forward(context, values);

				SafeGGmlTensor q = context.Reshape4d(qCur, qCur.Shape[0] / headers, headers, qCur.Shape[1], qCur.Shape[2]);
				SafeGGmlTensor k = context.Reshape4d(kCur, kCur.Shape[0] / headers, headers, kCur.Shape[1], kCur.Shape[2]);
				SafeGGmlTensor v = context.Reshape4d(vCur, vCur.Shape[0] / headers, headers, vCur.Shape[1], vCur.Shape[2]);

				q = context.Cont(context.Permute(q, 0, 2, 1, 3));
				k = context.Cont(context.Permute(k, 0, 2, 1, 3));
				v = context.Cont(context.Permute(v, 0, 2, 1, 3));

				// q * k
				SafeGGmlTensor qk = context.MulMat(k, q);

				SafeGGmlTensor kqScaled = context.Scale(qk, 1.0f / (float)Math.Sqrt(q.Shape[0]));

				SafeGGmlTensor kqSoftmax = context.Softmax(kqScaled);

				SafeGGmlTensor qkv = context.MulMat(kqSoftmax, context.Cont(context.Transpose(v)));

				SafeGGmlTensor qkvMerged = context.Cont(context.Transpose(qkv));
				qkvMerged = context.Cont(context.Permute(qkvMerged, 0, 2, 1, 3));
				qkvMerged = context.Reshape3d(qkvMerged, qkvMerged.Shape[0] * qkvMerged.Shape[1], qkvMerged.Shape[2], qkvMerged.Shape[3]);
				qkvMerged = outLinear.Forward(context, qkvMerged);

				return qkvMerged;

			}


		}


		public class MultiheadAttention
		{
			public long embed_dim = 768;
			public long n_head = 12;
			public bool useBias = true;

			public Linear q_proj { get; set; }
			public Linear k_proj { get; set; }
			public Linear v_proj { get; set; }
			public Linear out_proj { get; set; }

			public MultiheadAttention()
			{

			}

			public MultiheadAttention(Linear qLinear, Linear kLinear, Linear vLinear, Linear outLinear, long embed_dim, long n_head, bool bias = true)
			{
				this.q_proj = qLinear;
				this.k_proj = kLinear;
				this.v_proj = vLinear;
				this.out_proj = outLinear;

				this.embed_dim = embed_dim;
				this.n_head = n_head;
				this.useBias = bias;
			}

			public MultiheadAttention(SafeGGmlTensor qWeight, SafeGGmlTensor qBias, SafeGGmlTensor kWeight, SafeGGmlTensor kBias, SafeGGmlTensor vWeight, SafeGGmlTensor vBias, SafeGGmlTensor outWeight, SafeGGmlTensor outBias, long embed_dim, long n_head, bool bias = true)
			{
				this.q_proj = new Linear(qWeight, qBias);
				this.k_proj = new Linear(kWeight, kBias);
				this.v_proj = new Linear(vWeight, vBias);
				this.out_proj = new Linear(outWeight, outBias);

				this.embed_dim = embed_dim;
				this.n_head = n_head;
				this.useBias = bias;
			}


			public SafeGGmlTensor Forward(SafeGGmlContext ctx, SafeGGmlTensor x, bool mask = true)
			{
				long N = x.Shape[2];
				long n_token = x.Shape[1];
				long d_head = embed_dim / n_head;


				SafeGGmlTensor q = q_proj.Forward(ctx, x);
				q = ctx.Reshape4d(q, d_head, n_head, n_token, N);           // [N, n_token, n_head, d_head]
				q = ctx.Cont(ctx.Permute(q, 0, 2, 1, 3));                   // [N, n_head, n_token, d_head]
				q = ctx.Reshape3d(q, d_head, n_token, n_head * N);          // [N * n_head, n_token, d_head]

				SafeGGmlTensor k = k_proj.Forward(ctx, x);
				k = ctx.Reshape4d(k, d_head, n_head, n_token, N);           // [N, n_token, n_head, d_head]
				k = ctx.Cont(ctx.Permute(k, 0, 2, 1, 3));                   // [N, n_head, n_token, d_head]
				k = ctx.Reshape3d(k, d_head, n_token, n_head * N);          // [N * n_head, n_token, d_head]

				SafeGGmlTensor v = q_proj.Forward(ctx, x);
				v = ctx.Reshape4d(v, d_head, n_head, n_token, N);           // [N, n_token, n_head, d_head]
				v = ctx.Cont(ctx.Permute(v, 0, 2, 1, 3));                   // [N, n_head, n_token, d_head]
				v = ctx.Reshape3d(v, n_token, d_head, n_head * N);          // [N * n_head, n_token, d_head]

				SafeGGmlTensor kqv = Atten(ctx, q, k, v, mask, useCuda: true);              // [N * n_head, n_token, d_head]

				kqv = ctx.Reshape4d(kqv, d_head, n_token, n_head, N);
				kqv = ctx.Cont(ctx.Permute(kqv, 0, 2, 1, 3));               // [N, n_token, n_head, d_head]
				x = ctx.Reshape3d(kqv, d_head * n_head, n_token, N);        // [N, n_token, d_head * n_head]

				x = out_proj.Forward(ctx, x);                               // [N, n_token, embed_dim]
				return x;
			}

			private SafeGGmlTensor Atten(SafeGGmlContext ctx, SafeGGmlTensor q, SafeGGmlTensor k, SafeGGmlTensor v, bool masked = false, bool useCuda = false)
			{
				if (!useCuda)
				{
					SafeGGmlTensor mask = ctx.NewTensor2d(q.Type, q.Shape[2], q.Shape[1]);
					return ctx.FlashAttentionEx(q, k, v, mask, 1.0f / (float)Math.Sqrt(q.Shape[0]), 0);
				}
				else
				{
					float d_head = q.Shape[0];

					SafeGGmlTensor kq = ctx.MulMat(k, q);  // [N * n_head, n_token, n_k]
					kq = ctx.ScaleInplace(kq, (float)(1.0f / Math.Sqrt(d_head)));
					if (masked)
					{
						kq = ctx.DiagMaskInfInplace(kq, 0);
					}
					kq = ctx.SoftmaxInplace(kq);

					return ctx.MulMat(v, kq);  // [N * n_head, n_token, d_head]

				}
			}

		}
	}
}

