using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Converter.Abstractions
{
    public interface ISafeTensorConverter
    {
        void Convert(string safetensorsPath, string outputFileName, bool WriteToFileUsingStream = true);
    }
}
