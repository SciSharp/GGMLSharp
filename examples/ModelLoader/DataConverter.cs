using GGMLSharp;
using System.Runtime.InteropServices;

namespace ModelLoader
{
	public class DataConverter
	{
		public static void Bf16ToFp16Bytes(byte[] data)
		{
			for (int j = 0; j < data.Length; j += 2)
			{
				ushort data16 = (ushort)(data[j] | (data[j + 1] << 8));
				float data32 = Native.ggml_bf16_to_fp32(data16);
				data16 = Native.ggml_fp32_to_fp16(data32);
				byte[] bytes = BitConverter.GetBytes(data16);
				data[j] = bytes[0];
				data[j + 1] = bytes[1];
			}
		}

		public static void Bf16ToFp32Bytes(ref byte[] data)
		{
			Array.Resize(ref data, data.Length * 2);
			for (int j = data.Length / 4 - 1; j >= 0; j--)
			{
				ushort data16 = (ushort)(data[j * 2] | (data[j * 2 + 1] << 8));
				float data32 = Native.ggml_bf16_to_fp32(data16);
				byte[] bytes = BitConverter.GetBytes(data32);
				data[j * 4] = bytes[0];
				data[j * 4 + 1] = bytes[1];
				data[j * 4 + 2] = bytes[2];
				data[j * 4 + 3] = bytes[3];
			}
		}

		public static void Fp16ToFp32Bytes(ref byte[] data)
		{
			Array.Resize(ref data, data.Length * 2);
			for (int j = data.Length / 4 - 1; j >= 0; j--)
			{
				ushort data16 = (ushort)(data[j * 2] | (data[j * 2 + 1] << 8));
				float data32 = Native.ggml_fp16_to_fp32(data16);
				byte[] bytes = BitConverter.GetBytes(data32);
				data[j * 4] = bytes[0];
				data[j * 4 + 1] = bytes[1];
				data[j * 4 + 2] = bytes[2];
				data[j * 4 + 3] = bytes[3];
			}
		}

		public static void Fp16ToBf16Bytes(byte[] data)
		{
			for (int j = 0; j < data.Length; j += 2)
			{
				ushort data16 = (ushort)(data[j] | (data[j + 1] << 8));
				float data32 = Native.ggml_fp16_to_fp32(data16);
				data16 = Native.ggml_fp32_to_bf16(data32);
				byte[] bytes = BitConverter.GetBytes(data16);
				data[j] = bytes[0];
				data[j + 1] = bytes[1];
			}
		}

		public static void Fp32ToBf16Bytes(byte[] data)
		{
			for (int j = 0; j < data.Length / 4; j++)
			{
				float f32 = BitConverter.ToSingle(data, j * 4);
				ushort f16Data = Native.ggml_fp32_to_bf16(f32);
				byte[] bt = BitConverter.GetBytes(f16Data);
				data[j * 2] = bt[0];
				data[j * 2 + 1] = bt[1];
			}
			Array.Resize(ref data, data.Length / 2);
		}

		public static void Fp32ToFp16Bytes(byte[] data)
		{
			for (int j = 0; j < data.Length / 4; j++)
			{
				float f32 = BitConverter.ToSingle(data, j * 4);
				ushort f16Data = Native.ggml_fp32_to_fp16(f32);
				byte[] bt = BitConverter.GetBytes(f16Data);
				data[j * 2] = bt[0];
				data[j * 2 + 1] = bt[1];
			}
			Array.Resize(ref data, data.Length / 2);
		}
	}
}
