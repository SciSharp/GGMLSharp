namespace ModelLoader
{
	public interface IModelLoader
	{
		public List<Tensor> ReadTensorsInfoFromFile(string fileName);
		public byte[] ReadByteFromFile(Tensor tensor);
	}
}
