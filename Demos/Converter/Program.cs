namespace Converter
{
	internal class Program
	{
		static void Main(string[] args)
		{
			// Open_llama convert can work well, and Qwen convert is working now.
			// Tested Open_llama Model link is from https://huggingface.co/RiversHaveWings/open_llama_7b_safetensors
			// Tested Qwen model link is from https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
			// Tested LLama3 model link is from https://huggingface.co/bineric/NorskGPT-Llama3-8b/tree/main

			//new Safetensors.Qwen().Convert(@".\models\qwen", "qwen.gguf");
			//new Safetensors.Llama().Convert(@".\models\open_llama", "open_llama.gguf");
			//new Pickle.Llama3().Convert(@".\models\NorskGPT-Llama3-8b", "NorskGPT-Llama3-8b.gguf");
			
			new CommonTensorConverterDemo().ConvertToGguf(@".\Assets\model.pt", "model.gguf", CommonTensorConverterDemo.ModelType.Pickle, true);
			//new CommonTensorConverterDemo().ConvertToGguf(@".\Assets\taesd.safetensors", "taesd.gguf", CommonTensorConverterDemo.ModelType.Safetensors, false);

		}
	}
}
