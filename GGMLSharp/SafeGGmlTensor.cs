using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlTensor : SafeGGmlHandleBase
	{
		private ggml_tensor* tensor => (ggml_tensor*)handle;

		private bool IsInitialized => handle != IntPtr.Zero;

		public SafeGGmlTensor()
		{
			handle = IntPtr.Zero;
		}

		public SafeGGmlTensor(IntPtr intPtr)
		{
			this.handle = intPtr;
		}

		internal SafeGGmlTensor(ggml_tensor* tensor)
		{
			this.handle = (IntPtr)(tensor);
		}

		public SafeGGmlTensor(SafeGGmlContext context, Structs.GGmlType type, long[] shape)
		{
			this.handle = Native.ggml_new_tensor(context, type, shape.Length, shape).handle;
		}
		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public string Name
		{
			get { return Marshal.PtrToStringAnsi((IntPtr)tensor->name); }
			set { Native.ggml_set_name(this, value); }
		}

		public void FormatName(string name)
		{
			Native.ggml_format_name(this, name);
		}

		public string ViewName { get; set; }

		public long[] Shape
		{
			get
			{
				long[] shape = new long[4];
				shape[0] = tensor->ne[0];
				shape[1] = tensor->ne[1];
				shape[2] = tensor->ne[2];
				shape[3] = tensor->ne[3];
				return shape;
			}
		}

		public ulong[] Stride
		{
			get
			{
				ulong[] stride = new ulong[4];
				stride[0] = tensor->nb[0];
				stride[1] = tensor->nb[1];
				stride[2] = tensor->nb[2];
				stride[3] = tensor->nb[3];
				return stride;
			}
		}

		public Structs.GGmlType Type => (Structs.GGmlType)tensor->type;

		public Structs.GGmlBackendType Backend => (Structs.GGmlBackendType)tensor->backend;

		public IntPtr Data => tensor->data;

		public SafeGGmlTensor Grad => new SafeGGmlTensor((IntPtr)tensor->grad);

		public SafeGGmlTensor ViewSource => new SafeGGmlTensor(tensor->view_src);


		public Structs.GGmlOperation Operations
		{
			get { return (Structs.GGmlOperation)tensor->op; }
			set { tensor->op = (InternalStructs.ggml_op)value; }
		}


		public SafeGGmlTensor[] Sources
		{
			get
			{
				SafeGGmlTensor[] src = new SafeGGmlTensor[GGML_MAX_SRC];
				for (int i = 0; i < GGML_MAX_SRC; i++)
				{
					src[i] = new SafeGGmlTensor(new IntPtr(tensor->src[i]));
				}
				return src;
			}
		}

		public void SetData(byte[] data)
		{
			ThrowIfNotInitialized();
			if (tensor->data == IntPtr.Zero)
			{
				tensor->data = Marshal.AllocHGlobal(data.Length);
			}
			Marshal.Copy(data, 0, tensor->data, data.Length);
		}

		public void SetData(float[] data)
		{
			ThrowIfNotInitialized();
			if (tensor->data == IntPtr.Zero)
			{
				tensor->data = Marshal.AllocHGlobal(data.Length * sizeof(float));
			}
			Marshal.Copy(data, 0, tensor->data, data.Length);
		}

		public void SetData(int[] data)
		{
			ThrowIfNotInitialized();
			if (tensor->data == IntPtr.Zero)
			{
				tensor->data = Marshal.AllocHGlobal(data.Length * sizeof(float));
			}
			Marshal.Copy(data, 0, tensor->data, data.Length);
		}

		//public void SetData(float data, int index)
		//{
		//	ThrowIfNotInitialized();
		//	Native.ggml_set_f32_1d(this, index, data);
		//}

		public void SetData(float data, int ne0, int ne1 = 0, int ne2 = 0, int ne3 = 0)
		{
			ThrowIfNotInitialized();
			Native.ggml_set_f32_nd(this, ne0, ne1, ne2, ne3, data);
		}

		public float GetFloat(int n0 = 0, int n1 = 0, int n2 = 0, int n3 = 0)
		{
			return Native.ggml_get_f32_nd(this, n0, n1, n2, n3);
		}

		public float[] GetDataInFloats()
		{
			long length = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			float* f = Native.ggml_get_data_f32(this);
			float[] floats = new float[length];

			for (long i = 0; i < length; i++)
			{
				floats[i] = f[i];
			}
			return floats;
		}

		public byte[] GetData()
		{
			ulong size = ElementsSize * (ulong)ElementsCount;
			IntPtr ptr = (IntPtr)Native.ggml_get_data(this);
			byte[] bytes = new byte[size];
			Marshal.Copy(ptr, bytes, 0, (int)size);
			return bytes;
		}

		public void SetBackend(IntPtr data, ulong offset = 0, ulong size = 0)
		{
			ThrowIfNotInitialized();
			size = size == 0 ? ElementsSize * (ulong)ElementsCount : size;
			Native.ggml_backend_tensor_set(this, data, offset, size);
		}

		public void SetDataToBackend()
		{
			ThrowIfNotInitialized();
			if (Data == IntPtr.Zero)
			{
				throw new ArgumentNullException("data is null");
			}
			SetBackend(GetData());
		}

		public void SetBackend(Array data, ulong offset = 0, ulong size = 0)
		{
			ThrowIfNotInitialized();

			IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
			size = size == 0 ? ElementsSize * (ulong)ElementsCount : size;
			int sz = Marshal.SizeOf(data.GetType().GetElementType());
			if ((ulong)(sz * data.Length) != size)
			{
				throw new ArgumentOutOfRangeException("data size is not fit");
			}
			Native.ggml_backend_tensor_set(this, ptr, offset, size);
		}



		public long ElementsCount => Native.ggml_nelements(this);
		public ulong ElementsSize => Native.ggml_element_size(this);

		public byte[] GetBackend()
		{
			ThrowIfNotInitialized();
			ulong size = (ulong)ElementsCount * ElementsSize;
			IntPtr ptr = Marshal.AllocHGlobal((int)size);
			Native.ggml_backend_tensor_get(this, ptr, 0, size);
			byte[] bytes = new byte[size];
			Marshal.Copy(ptr, bytes, 0, (int)size);
			return bytes;
		}

		public void SetInput()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_input(this);
		}

		public void SetOutput()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_output(this);
		}

		public void SetRandomTensorInFloat(float max, float min)
		{
			ThrowIfNotInitialized();
			Random random = new Random();

			long size = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			for (long i = 0; i < size; i++)
			{
				float f = (float)random.NextDouble() * (max - min) + min;
				SetData(f, (int)i);
			}
		}


		public void SetBackendRandomTensorInFloat(float max, float min)
		{
			ThrowIfNotInitialized();
			Random random = new Random();

			long size = Shape[0] * Shape[1] * Shape[2] * Shape[3];
			float[] floats = new float[size];
			for (long i = 0; i < size; i++)
			{
				floats[i] = (float)random.NextDouble() * (max - min) + min;
			}
			SetBackend(floats);
		}

		public void SetZero()
		{
			ThrowIfNotInitialized();
			Native.ggml_set_zero(this);
		}

		public bool AreSameShape(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			return Native.ggml_are_same_shape(this, tensor);
		}

		public bool IsContiguous()
		{
			ThrowIfNotInitialized();
			return Native.ggml_is_contiguous(this);
		}

		public bool IsQuantized()
		{
			ThrowIfNotInitialized();
			return Native.ggml_is_quantized(this.Type);
		}




	}
}
