using System.Text.RegularExpressions;

namespace Converter.CommonLib
{
	internal class DataTrans
	{
		internal static string TensorNameTransToGgufName(string inputTensorName)
		{
			if (inputTensorName == "lm_head.Weight")
			{
				return "output.Weight";
			}
			else if (inputTensorName == "model.embed_tokens.Weight")
			{
				return "token_embd.Weight";
			}
			else if (inputTensorName == "model.norm.Weight")
			{
				return "output_norm.Weight";
			}
			else if (Regex.IsMatch(inputTensorName, @"model.layers.(\d+).input_layernorm.Weight"))
			{
				string num = new Regex(@"model.layers.(\d+).input_layernorm.Weight").Match(inputTensorName).Groups[1].Value;
				return $"blk.{num}.attn_norm.Weight";
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
