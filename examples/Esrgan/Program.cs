namespace Esrgan
{
	internal class Program
	{
		static void Main(string[] args)
		{
			List<PickleLoader.CommonTensor> tensors = PickleLoader.ReadTensorInfoFromFile(@".\Assets\RealESRGAN_x4plus_anime_6B.pth");
			
		}
	}
}
