using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlBackendBuffer : SafeGGmlHandleBase
	{
		private ggml_backend_buffer* ggml_backend_buffer => (ggml_backend_buffer*)handle;
		
		public SafeGGmlBackendBuffer()
		{
			this.handle = IntPtr.Zero;
		}

		internal SafeGGmlBackendBuffer(ggml_backend_buffer* ggml_backend_buffer)
		{
			handle = (IntPtr)ggml_backend_buffer;
		}

		public SafeGGmlBackendBufferType BufferType => Native.ggml_backend_buffer_get_type(this);

		public void Free()
		{
			Native.ggml_backend_buffer_free(this);
		}

	}
}
