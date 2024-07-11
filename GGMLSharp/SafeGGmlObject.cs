using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlObject : SafeGGmlHandleBase
	{
		private ggml_object* ggml_object => (ggml_object*)handle;

		public SafeGGmlObject()
		{
			this.handle = IntPtr.Zero;
		}

		internal SafeGGmlObject(ggml_object* @object)
		{
			this.handle = (IntPtr)@object;
		}

		public ulong Offset => ggml_object->offs;
		public ulong Size => ggml_object->size;

		public SafeGGmlObject Next => new SafeGGmlObject(ggml_object->next);

		public Structs.GGmlObjectType Type => (Structs.GGmlObjectType)ggml_object->type;

		//public fixed byte padding[4];

	}
}
