﻿using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlContext : SafeGGmlHandleBase
	{
		private ggml_context* context => (ggml_context*)handle;

		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public SafeGGmlContext(IntPtr buffer, ulong Size = 10 * 1024 * 1024, bool NoAllocateMemory = false)
		{
			ggml_init_params @params = new ggml_init_params
			{
				no_alloc = NoAllocateMemory,
				mem_buffer = buffer,
				mem_size = Size,
			};
			handle = Native.ggml_init(@params);
		}

		public SafeGGmlContext(byte[] buffer, bool NoAllocateMemory = false)
		{
			IntPtr des = Marshal.AllocHGlobal(buffer.Length);
			Marshal.Copy(buffer, 0, des, buffer.Length);
			ggml_init_params @params = new ggml_init_params
			{
				no_alloc = NoAllocateMemory,
				mem_buffer = des,
				mem_size = (ulong)buffer.Length,
			};
			Marshal.Copy(des, buffer, 0, buffer.Length);
			handle = Native.ggml_init(@params);
		}

		public SafeGGmlContext()
		{
			ggml_init_params @params = new ggml_init_params
			{
				mem_buffer = IntPtr.Zero,
				mem_size = 10 * 1024 * 1024,
				no_alloc = false,
			};
			handle = Native.ggml_init(@params);
		}

		internal void SetContext(ggml_context* context)
		{
			this.handle = (IntPtr)context;
		}


		public IntPtr MemoryBuff => context->mem_buffer;
		public ulong MemorySize => context->mem_size;
		public bool NoAlloc => Convert.ToBoolean(context->no_alloc);
		public bool NoAllocSave => Convert.ToBoolean(context->no_alloc_save);
		public int ObjectsCount => context->n_objects;
		public bool MemoryBufferOwned => Convert.ToBoolean(context->mem_buffer_owned);

		public SafeGGmlObject ObjectsBegin => new SafeGGmlObject(context->objects_begin);
		public SafeGGmlObject ObjectsEnd => new SafeGGmlObject(context->objects_end);

		public SafeGGmlContext(IntPtr intPtr)
		{
			this.handle = intPtr;
		}

		public int ObjectCount()
		{
			ThrowIfNotInitialized();
			return context->n_objects;
		}

		public void Free()
		{
			if (IsInitialized)
			{
				Native.ggml_free(handle);
				handle = IntPtr.Zero;
			}
		}

		private bool IsInitialized => handle != IntPtr.Zero;

		public SafeGGmlTensor Transpose(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			return Native.ggml_transpose(this, tensor);
		}

		public static int GetPad(int x, int y)
		{
			return Native.GGML_PAD(x, y);
		}

		public SafeGGmlTensor NewTensor1d(Structs.GGmlType type, long ne0)
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_tensor_1d(this, type, ne0);
		}

		public SafeGGmlTensor NewTensor2d(Structs.GGmlType type, long ne0, long ne1)
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_tensor_2d(this, type, ne0, ne1);
		}

		public SafeGGmlTensor NewTensor3d(Structs.GGmlType type, long ne0, long ne1, long ne2)
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_tensor_3d(this, type, ne0, ne1, ne2);
		}

		public SafeGGmlTensor NewTensor4d(Structs.GGmlType type, long ne0, long ne1, long ne2, long ne3)
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_tensor_4d(this, type, ne0, ne1, ne2, ne3);
		}

		public SafeGGmlTensor NewTensor(Structs.GGmlType type, long[] shape)
		{
			ThrowIfNotInitialized();
			if (shape.Length < 1 || shape.Length > 4)
			{
				throw new ArgumentOutOfRangeException("Shape is not support");
			}
			return Native.ggml_new_tensor(this, type, shape.Length, shape);
		}

		public SafeGGmlBackendBuffer BackendAllocContextTensors(SafeGGmlBackend backend)
		{
			ThrowIfNotInitialized();
			return Native.ggml_backend_alloc_ctx_tensors(this, backend);
		}

		public SafeGGmlTensor MulMat(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			ThrowIfNotInitialized();
			return Native.ggml_mul_mat(this, a, b);
		}

		public void SetParam(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			Native.ggml_set_param(this, tensor);
		}

		public SafeGGmlTensor GetTensor(string name)
		{
			ThrowIfNotInitialized();
			return Native.ggml_get_tensor(this, name);
		}

		public SafeGGmlTensor Add(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			ThrowIfNotInitialized();
			return Native.ggml_add(this, a, b);
		}

		public SafeGGmlTensor Gelu(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_gelu(this, a);
		}

		public SafeGGmlTensor Relu(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_relu(this, a);
		}


		public SafeGGmlTensor Cont(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_cont(this, a);
		}

		public SafeGGmlTensor Reshape1d(SafeGGmlTensor a, long ne0)
		{
			ThrowIfNotInitialized();
			return Native.ggml_reshape_1d(this, a, ne0);
		}

		public SafeGGmlTensor Reshape2d(SafeGGmlTensor a, long ne0, long ne1)
		{
			ThrowIfNotInitialized();
			return Native.ggml_reshape_2d(this, a, ne0, ne1);
		}

		public SafeGGmlTensor Reshape3d(SafeGGmlTensor a, long ne0, long ne1, long ne2)
		{
			ThrowIfNotInitialized();
			return Native.ggml_reshape_3d(this, a, ne0, ne1, ne2);
		}
		public SafeGGmlTensor Reshape4d(SafeGGmlTensor a, long ne0, long ne1, long ne2, long ne3)
		{
			ThrowIfNotInitialized();
			return Native.ggml_reshape_4d(this, a, ne0, ne1, ne2, ne3);
		}

		public SafeGGmlTensor Normal(SafeGGmlTensor a, float eps)
		{
			ThrowIfNotInitialized();
			return Native.ggml_norm(this, a, eps);
		}

		public SafeGGmlTensor Mul(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			ThrowIfNotInitialized();
			return Native.ggml_mul(this, a, b);
		}

		public SafeGGmlTensor Pool1d(SafeGGmlTensor a, Structs.GGmlOpPool op, int k0, int s0, int p0)
		{
			ThrowIfNotInitialized();
			return Native.ggml_pool_1d(this, a, (ggml_op_pool)op, k0, s0, p0);
		}

		public SafeGGmlTensor Pool2d(SafeGGmlTensor a, Structs.GGmlOpPool op = Structs.GGmlOpPool.GGML_OP_POOL_MAX, int k0 = 2, int k1 = 2, int s0 = 2, int s1 = 2, int p0 = 0, int p1 = 0)
		{
			ThrowIfNotInitialized();
			return Native.ggml_pool_2d(this, a, (ggml_op_pool)op, k0, k1, s0, s1, p0, p1);
		}

		public SafeGGmlTensor SoftMax(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_soft_max(this, a);
		}

		public SafeGGmlTensor Sub(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			ThrowIfNotInitialized();
			return Native.ggml_sub(this, a, b);
		}

		public SafeGGmlTensor Sum(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_sum(this, a);
		}

		public SafeGGmlTensor Sqr(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_sqr(this, a);
		}

		public SafeGGmlTensor Sqrt(SafeGGmlTensor a)
		{
			ThrowIfNotInitialized();
			return Native.ggml_sqrt(this, a);
		}

		public SafeGGmlGraph NewGraph()
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_graph(this);
		}

		public SafeGGmlGraph CustomNewGraph(ulong size = GGML_DEFAULT_GRAPH_SIZE, bool grads = true)
		{
			ThrowIfNotInitialized();
			return Native.ggml_new_graph_custom(this, size, grads);
		}

		public Structs.OptimizationResult Optimizer(Structs.OptimizerParameters @params, SafeGGmlTensor f)
		{
			ThrowIfNotInitialized();
			return Native.ggml_opt(this, @params, f);
		}

		public static Structs.OptimizerParameters GetDefaultOptimizerParams(Structs.OptimizerType type)
		{
			return Native.ggml_opt_default_params(type);
		}

		public static Structs.OptimizationResult OptimizerWithDefaultGGmlContext(Structs.OptimizerParameters @params, SafeGGmlTensor f)
		{
			return Native.ggml_opt(IntPtr.Zero, @params, f);
		}

		public SafeGGmlTensor Conv2d(SafeGGmlTensor a, SafeGGmlTensor b, int s0 = 1, int s1 = 1, int p0 = 0, int p1 = 0, int d0 = 1, int d1 = 1)
		{
			ThrowIfNotInitialized();
			return Native.ggml_conv_2d(this, a, b, s0, s1, p0, p1, d0, d1);
		}

		public SafeGGmlTensor CrossEntropyLoss(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_cross_entropy_loss(this, a, b);
		}

		public SafeGGmlTensor Scale(SafeGGmlTensor tensor, float s)
		{
			return Native.ggml_scale(this, tensor, s);
		}

		public SafeGGmlTensor MapCustom1(SafeGGmlTensor a, Structs.Custom1OpDelegate fun, int taskCount, IntPtr userdata)
		{
			return Native.ggml_map_custom1(this, a, fun, taskCount, userdata);
		}

		public SafeGGmlTensor MapCustom2(SafeGGmlTensor a, SafeGGmlTensor b, Structs.Custom2OpDelegate fun, int taskCount, IntPtr userdata)
		{
			return Native.ggml_map_custom2(this, a, b, fun, taskCount, userdata);
		}

		public SafeGGmlTensor MapCustom3(SafeGGmlTensor a, SafeGGmlTensor b, SafeGGmlTensor c, Structs.Custom3OpDelegate fun, int taskCount, IntPtr userdata)
		{
			return Native.ggml_map_custom3(this, a, b, c, fun, taskCount, userdata);
		}

		public SafeGGmlTensor View1d(SafeGGmlTensor tensor, long ne0, ulong offset)
		{
			return Native.ggml_view_1d(this, tensor, ne0, offset);
		}

		public SafeGGmlTensor View2d(SafeGGmlTensor tensor, long ne0, long ne1, ulong nb1, ulong offset)
		{
			return Native.ggml_view_2d(this, tensor, ne0, ne1, nb1, offset);
		}

		public SafeGGmlTensor View3d(SafeGGmlTensor tensor, long ne0, long ne1, long ne2, ulong nb1, ulong nb2, ulong offset)
		{
			return Native.ggml_view_3d(this, tensor, ne0, ne1, ne2, nb1, nb2, offset);
		}

		public SafeGGmlTensor View4d(SafeGGmlTensor tensor, long ne0, long ne1, long ne2, long ne3, ulong nb1, ulong nb2, ulong nb3, ulong offset)
		{
			return Native.ggml_view_4d(this, tensor, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset);
		}

		public SafeGGmlTensor ViewTensor(SafeGGmlTensor tensor)
		{
			return Native.ggml_view_tensor(this, tensor);
		}

		public SafeGGmlTensor Copy(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_cpy(this, a, b);

		}

		public SafeGGmlTensor Permute(SafeGGmlTensor tensor, int axis0, int axis1, int axis2, int axis3)
		{
			return Native.ggml_permute(this, tensor, axis0, axis1, axis2, axis3);
		}

		public SafeGGmlTensor AddInplace(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_add_inplace(this, a, b);
		}

		public SafeGGmlTensor Repeat(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_repeat(this, a, b);
		}

		public SafeGGmlTensor ScaleInplace(SafeGGmlTensor tensor, float s)
		{
			return Native.ggml_scale_inplace(this, tensor, s);
		}

		public SafeGGmlTensor SoftmaxInplace(SafeGGmlTensor tensor)
		{
			return Native.ggml_soft_max_inplace(this, tensor);
		}

		public SafeGGmlTensor ReluInplace(SafeGGmlTensor tensor)
		{
			return Native.ggml_relu_inplace(this, tensor);
		}

		public SafeGGmlTensor NormalInplace(SafeGGmlTensor tensor, float eps)
		{
			return Native.ggml_norm_inplace(this, tensor, eps);
		}

		public SafeGGmlTensor ConvTranspose2dP0(SafeGGmlTensor a, SafeGGmlTensor b, int stride)
		{
			return Native.ggml_conv_transpose_2d_p0(this, a, b, stride);
		}

		public SafeGGmlTensor GeluInplace(SafeGGmlTensor tensor)
		{
			return Native.ggml_elu_inplace(this, tensor);
		}

		public SafeGGmlTensor Conv2dSkP0(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_conv_2d_sk_p0(this, a, b);
		}

		public SafeGGmlTensor GetRelPos(SafeGGmlTensor tensor, int qh, int kh)
		{
			return Native.ggml_get_rel_pos(this, tensor, qh, kh);
		}

		public SafeGGmlTensor AddRelPosInplace(SafeGGmlTensor tensor, SafeGGmlTensor pw, SafeGGmlTensor ph)
		{
			return Native.ggml_add_rel_pos_inplace(this, tensor, pw, ph);
		}

		public SafeGGmlTensor WinUnpart(SafeGGmlTensor tensor, int w0, int h0, int w)
		{
			return Native.ggml_win_unpart(this, tensor, w0, h0, w);
		}

		public SafeGGmlTensor Conv2dS1Ph(SafeGGmlTensor a, SafeGGmlTensor b)
		{
			return Native.ggml_conv_2d_s1_ph(this, a, b);
		}

		public SafeGGmlTensor WinPart(SafeGGmlTensor tensor, int w)
		{
			return Native.ggml_win_part(this, tensor, w);
		}



	}
}