
namespace Converter.Abstractions
{
	public interface ISafeTensorConverter
    {
        void Convert(string safetensorsPath, string outputFileName, bool WriteToFileUsingStream = true);
    }
}
