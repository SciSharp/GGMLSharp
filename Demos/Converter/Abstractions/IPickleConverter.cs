namespace Converter.Abstractions
{
	public interface IPickleConverter
	{
		void Convert(string picklePath, string outputFileName, bool WriteToFileUsingStream = true);
	}
}
