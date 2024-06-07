using GGMLSharp;
using System.Text.RegularExpressions;

namespace Converter.CommonLib
{
	internal class DataTrans
	{
		internal static byte[] Bf16ToFp16Bytes(byte[] bf16Bytes)
		{
			for (int j = 0; j < bf16Bytes.Length; j += 2)
			{
				ushort data16 = (ushort)(bf16Bytes[j] | (bf16Bytes[j + 1] << 8));
				float data32 = Native.ggml_bf16_to_fp32(data16);
				data16 = Native.ggml_fp32_to_fp16(data32);
				byte[] bytes = BitConverter.GetBytes(data16);
				bf16Bytes[j] = bytes[0];
				bf16Bytes[j + 1] = bytes[1];
			}
			return bf16Bytes;
		}

		internal static byte[] Bf16ToF32Bytes(byte[] bf16Bytes)
		{
			byte[] f32bytes = new byte[bf16Bytes.Length * 2];
			for (int j = 0; j < bf16Bytes.Length / 2; j++)
			{
				ushort data16 = (ushort)(bf16Bytes[j * 2] | (bf16Bytes[j * 2 + 1] << 8));
				float data32 = Native.ggml_bf16_to_fp32(data16);
				byte[] bytes = BitConverter.GetBytes(data32);
				f32bytes[j * 4] = bytes[0];
				f32bytes[j * 4 + 1] = bytes[1];
				f32bytes[j * 4 + 2] = bytes[2];
				f32bytes[j * 4 + 3] = bytes[3];
			}
			return f32bytes;
		}

		internal static byte[] Fp16ToF32Bytes(byte[] fp16Bytes)
		{
			byte[] f32bytes = new byte[fp16Bytes.Length * 2];
			for (int j = 0; j < fp16Bytes.Length / 2; j++)
			{
				ushort data16 = (ushort)(fp16Bytes[j * 2] | (fp16Bytes[j * 2 + 1] << 8));
				float data32 = Native.ggml_fp16_to_fp32(data16);
				byte[] bytes = BitConverter.GetBytes(data32);
				f32bytes[j * 4] = bytes[0];
				f32bytes[j * 4 + 1] = bytes[1];
				f32bytes[j * 4 + 2] = bytes[2];
				f32bytes[j * 4 + 3] = bytes[3];
			}
			return f32bytes;
		}

		internal static string TensorNameTransToGgufName(string inputTensorName)
		{
			if (inputTensorName == "lm_head.weight")
			{
				return "output.weight";
			}
			else if (inputTensorName == "model.embed_tokens.weight")
			{
				return "token_embd.weight";
			}
			else if (inputTensorName == "model.norm.weight")
			{
				return "output_norm.weight";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).input_layernorm.weight"))
			{
				string num = new Regex(@"model.layers.(\d+).input_layernorm.weight").Match(inputTensorName).Groups[1].Value;
				return $"blk.{num}.attn_norm.weight";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).mlp.(\w+)_proj.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).mlp.(\w+)_proj.(\w+)").Match(inputTensorName);
				return $"blk.{match.Groups[1]}.ffn_{match.Groups[2]}.{match.Groups[3]}";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).post_attention_layernorm.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).post_attention_layernorm.(\w+)").Match(inputTensorName);
				return $"blk.{match.Groups[1]}.ffn_norm.{match.Groups[2]}";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).self_attn.([kqvo])_proj.(\w+)"))
			{
				Match match = new Regex(@"model.layers.(\d+).self_attn.([kqvo])_proj.(\w+)").Match(inputTensorName);
				if (match.Groups[2].Value == "o")
				{
					return $"blk.{match.Groups[1]}.attn_output.{match.Groups[3]}";
				}
				return $"blk.{match.Groups[1]}.attn_{match.Groups[2]}.{match.Groups[3]}";
			}

			else
			{
				return inputTensorName;
			}
		}

	}
}
