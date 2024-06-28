using ModelLoader;

namespace Esrgan
{
	internal class Program
	{
		static void Main(string[] args)
		{
			IModelLoader modelLoader = new PickleLoader();
			List<Tensor> tensors = modelLoader.ReadTensorsInfoFromFile(@".\Assets\RealESRGAN_x4plus_anime_6B.pth");
			
		}
	}
}
