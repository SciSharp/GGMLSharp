using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlGraphAllocr : SafeGGmlHandleBase
	{
		private ggml_gallocr* allocr => (ggml_gallocr*)handle;

		public int BuffersCount => allocr->n_buffers;
		public int LeafsCount => allocr->n_leafs;
		public int NodesCount => allocr->n_nodes;


		public SafeGGmlBackendBufferType[] BufferTypes
		{
			get
			{
				SafeGGmlBackendBufferType[] bufferTypes = new SafeGGmlBackendBufferType[BuffersCount];
				for (int i = 0; i < BuffersCount; i++)
				{
					bufferTypes[i] = new SafeGGmlBackendBufferType(allocr->bufts[i]);
				}
				return bufferTypes;
			}
		}

		public SafeGGmlGraphAllocr(SafeGGmlBackendBufferType type)
		{
			this.handle = (IntPtr)Native.ggml_gallocr_new(type);
		}

		public SafeGGmlGraphAllocr()
		{
			this.handle = (IntPtr)Native.ggml_gallocr_new(Native.ggml_backend_cpu_buffer_type());
		}

		public SafeGGmlBackendBuffer[] Buffers
		{
			get
			{
				SafeGGmlBackendBuffer[] buffers = new SafeGGmlBackendBuffer[BuffersCount];
				for (int i = 0; i < BuffersCount; i++)
				{
					buffers[i] = new SafeGGmlBackendBuffer(allocr->buffers[i]);
				}
				return buffers;
			}
		}

		public bool Reserve(SafeGGmlGraph graph)
		{
			return Native.ggml_gallocr_reserve(this, graph);
		}

		public ulong GetBufferSize(int index)
		{
			return Native.ggml_gallocr_get_buffer_size(this, index);
		}

		public void Free()
		{
			Native.ggml_gallocr_free(handle);
			handle = IntPtr.Zero;
		}
	}
}
