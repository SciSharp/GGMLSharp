using static GGMLSharp.Structs;

namespace ModelLoader
{
	public class Tensor
	{
		public string Name { get; set; }
		public ggml_type Type { get; set; } = ggml_type.GGML_TYPE_F16;
		public List<long> Shape { get; set; } = new List<long>();
		public List<ulong> Stride { get; set; } = new List<ulong>();
		public string DataNameInZipFile { get; set; }
		public string FileName { get; set; }
		public List<ulong> Offset { get; set; } = new List<ulong>();
		public long BodyPosition { get; set; }

	}
}
