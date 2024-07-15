using System.Collections.Generic;

namespace ModelLoader
{
	public interface IModelLoader
	{
		List<Tensor> ReadTensorsInfoFromFile(string fileName);
		byte[] ReadByteFromFile(Tensor tensor);
	}
}
