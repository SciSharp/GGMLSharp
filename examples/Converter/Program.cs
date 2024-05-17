using System.Net.Http.Headers;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Converter
{
	internal class Program
	{
		static void Main(string[] args)
		{
			new SafeTensorReader().ConvertSafetensorsToGguf(@".\Assets\taesd.safetensors", @".\Assets\taesd.gguf");
		}
	}
}
