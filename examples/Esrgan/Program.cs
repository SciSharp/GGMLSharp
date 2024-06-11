namespace Esrgan
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Console.WriteLine("Hello, World!");
			List<PickleLoader.CommonTensor> tensors = PickleLoader.ReadTensorInfoFromFile(@"D:\DeepLearning\llama\ggml\examples\sam\sam_vit_b_01ec64.pth");
		

		}
	}
}
