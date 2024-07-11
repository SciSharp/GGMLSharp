using static GGMLSharp.Structs;

namespace GGMLSharp
{
	public class Common
	{
		public static ulong GetGGmlTypeSize(GGmlType type)
		{
			return Native.ggml_type_size(type);
		}
		public static ulong GetGGufTypeSize(GGufType type)
		{
			switch (type)
			{
				case GGufType.GGUF_TYPE_UINT8:
				case GGufType.GGUF_TYPE_INT8:
				case GGufType.GGUF_TYPE_BOOL:
					return 1;
				case GGufType.GGUF_TYPE_UINT16:
				case GGufType.GGUF_TYPE_INT16:
					return 2;
				case GGufType.GGUF_TYPE_UINT32:
				case GGufType.GGUF_TYPE_INT32:
				case GGufType.GGUF_TYPE_FLOAT32:
					return 4;

				case GGufType.GGUF_TYPE_UINT64:
				case GGufType.GGUF_TYPE_INT64:
				case GGufType.GGUF_TYPE_FLOAT64:
					return 8;
				case GGufType.GGUF_TYPE_ARRAY:
					return 0; // undefined
				case GGufType.GGUF_TYPE_STRING:
					return 16;
				default:
					return 0;
			}

		}

		public static ulong TensorOverheadLength => Native.ggml_tensor_overhead();
		public static ulong GraphOverheadLength => Native.ggml_graph_overhead();

		public static bool HasCuda => Native.ggml_cpu_has_cuda();



	}

}
