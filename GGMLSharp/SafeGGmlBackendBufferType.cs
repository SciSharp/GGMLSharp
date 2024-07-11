using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlBackendBufferType : SafeGGmlHandleBase
	{
		private ggml_backend_buffer_type* ggml_backend_buffer_type => (ggml_backend_buffer_type*)handle;

		internal SafeGGmlBackendBufferType(ggml_backend_buffer_type* buffer_type)
		{
			this.handle = (IntPtr)buffer_type;
		}

		public SafeGGmlBackendBufferType()
		{
			this.handle = IntPtr.Zero;
		}

		public IntPtr Context => ggml_backend_buffer_type->context;
		internal ggml_backend_buffer_type_i TypeInterface => ggml_backend_buffer_type->iface;

	}
}
