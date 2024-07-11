using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGufTensorInfo
	{
		private gguf_tensor_info info;

		public SafeGGufTensorInfo()
		{

		}

		internal SafeGGufTensorInfo(gguf_tensor_info info)
		{
			this.info = info;
		}

		public string Name => Marshal.PtrToStringAnsi(info.name.data);
		public ulong DimsCount => info.n_dims;
		public ulong[] Shape
		{
			get
			{
				ulong[] shape = new ulong[DimsCount];
				for (ulong i = 0; i < DimsCount; i++)
				{
					shape[i] = info.ne[i];
				}
				return shape;
			}
		}

		public ulong Offset => info.offset;
		public ulong Size => info.size;
		public Structs.GGmlType Type => (Structs.GGmlType)info.type;
		public IntPtr Data => info.data;
	}
}
