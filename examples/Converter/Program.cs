namespace Converter
{
	internal class Program
	{
		static void Main(string[] args)
		{
			// Open_llama convert can work well, and Qwen convert is working now.
			// Tested Open_llama Model link is from https://huggingface.co/RiversHaveWings/open_llama_7b_safetensors
			// Tested Qwen model link is from https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat
			// Phi can be converted by the same way as LLama, but have to trans the name of the model, not all weights contain in current model.
			// Ckpt cannot work, it's a different format.

			//new Safetensors.Llama().Convert(@".\models\open_llama", "open_llama.gguf");
			new Safetensors.Qwen().Convert(@".\models\qwen\1_5", "qwen.gguf");
			//new CommonSafeTensorReaderDemo().ConvertSafetensorsToGguf(@".\Assets\taesd.safetensors", "q.gguf");
		}
	}
}
