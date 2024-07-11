using System;
using System.Runtime.InteropServices;
using System.Text;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGufContext : SafeGGmlHandleBase, IDisposable
	{
		private gguf_context* gguf_context => (gguf_context*)handle;
		private bool IsInitialized => handle != IntPtr.Zero;
		public bool IsHeaderMagicMatch => Marshal.PtrToStringAnsi((IntPtr)gguf_context->header.magic).Contains("GGUF");

		public ulong TensorsCount => gguf_context->header.n_tensors;
		public uint Version => gguf_context->header.version;
		public ulong KeyValuesCount => gguf_context->header.n_kv;

		public ulong Alignment => gguf_context->alignment;

		public SafeGGufContext()
		{
			this.handle = IntPtr.Zero;
		}

		public static SafeGGufContext Initialize()
		{
			return Native.gguf_init_empty();
		}

		public static SafeGGufContext InitFromFile(string fileName, SafeGGmlContext ggmlContext, bool noAlloc)
		{
			return Native.gguf_init_from_file(fileName, ggmlContext, noAlloc);
		}

		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public SafeGGufKeyValue[] KeyValues
		{
			get
			{
				SafeGGufKeyValue[] kvs = new SafeGGufKeyValue[KeyValuesCount];
				for (ulong i = 0; i < KeyValuesCount; i++)
				{
					gguf_kv kv = Marshal.PtrToStructure<gguf_kv>(gguf_context->kv + (int)i * sizeof(gguf_kv));
					kvs[i] = new SafeGGufKeyValue(kv);
				}
				return kvs;
			}
		}

		public SafeGGufTensorInfo[] GGufTensorInfos
		{
			get
			{
				SafeGGufTensorInfo[] infos = new SafeGGufTensorInfo[TensorsCount];
				for (ulong i = 0; i < TensorsCount; i++)
				{
					//gguf_tensor_info info = Marshal.PtrToStructure<gguf_tensor_info>(gguf_context->infos + (int)i * sizeof(gguf_tensor_info));
					infos[i] = new SafeGGufTensorInfo(gguf_context->infos[i]);
				}
				return infos;
			}
		}

		public void SetValueString(string key, string value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_str(this, key, value);

		}

		public void SetValueBool(string key, bool value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_bool(this, key, value);

		}

		public void SetValueFloat(string key, float value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_f32(this, key, value);
		}

		public void SetValueFloat64(string key, double value)
		{
			if (IsInitialized)
			{
				Native.gguf_set_val_f64(this, key, value);
			}
		}

		public void SetValueInt16(string key, Int16 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i16(this, key, value);
		}

		public void SetValueInt32(string key, int value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i32(this, key, value);

		}

		public void SetValueInt64(string key, Int64 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i64(this, key, value);

		}

		public void SetValueInt8(string key, sbyte value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_i8(this, key, value);
		}


		public void SetValueUInt8(string key, byte value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u8(this, key, value);
		}

		public void SetValueUInt16(string key, UInt16 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u16(this, key, value);
		}

		public void SetValueUInt32(string key, uint value)
		{
			if (IsInitialized)
			{
				Native.gguf_set_val_u32(this, key, value);
			}
		}

		public void SetValueUInt64(string key, UInt64 value)
		{
			ThrowIfNotInitialized();
			Native.gguf_set_val_u64(this, key, value);
		}

		public void SetValueArrayData(string key, Structs.GGufType type, Array data)
		{
			ThrowIfNotInitialized();
			if (type == Structs.GGufType.GGUF_TYPE_ARRAY)
			{
				throw new ArgumentException("Array not support");
			}
			else if (type == Structs.GGufType.GGUF_TYPE_STRING)
			{
				gguf_str[] ggufStrs = new gguf_str[data.Length];
				for (int i = 0; i < data.Length; i++)
				{
					ggufStrs[i] = new gguf_str { data = StringToCoTaskMemUTF8((string)data.GetValue(i)), n = (ulong)data.GetValue(i).ToString().Length };
				}
				IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(ggufStrs, 0);
				Native.gguf_set_arr_data(this, key, type, ptr, data.Length);
			}
			else
			{
				IntPtr ptr = Marshal.UnsafeAddrOfPinnedArrayElement(data, 0);
				Native.gguf_set_arr_data(this, key, type, ptr, data.Length);
			}
		}

		public void SetValueArrayString(string key, string[] strs)
		{
			ThrowIfNotInitialized();
			IntPtr[] dataPtrs = new IntPtr[strs.Length];
			for (int i = 0; i < strs.Length; i++)
			{
				dataPtrs[i] = StringToCoTaskMemUTF8(strs[i]);
			}
			Native.gguf_set_arr_str(this, key, dataPtrs, strs.Length);
		}

		public void Free()
		{
			if (IsInitialized)
			{
				Native.gguf_free(handle);
				handle = IntPtr.Zero;
			}
		}

		public static unsafe IntPtr StringToCoTaskMemUTF8(string s)
		{
			if (s is null)
			{
				return IntPtr.Zero;
			}
			int nb = Encoding.UTF8.GetMaxByteCount(s.Length);
			IntPtr ptr = Marshal.AllocCoTaskMem(checked(nb + 1));
			byte* pbMem = (byte*)ptr;
			char[] chars = s.ToCharArray();
			Marshal.Copy(chars, 0, ptr, chars.Length);
			fixed (char* chr = chars)
			{
				int nbWritten = Encoding.UTF8.GetBytes(chr, chars.Length, pbMem, nb);
				pbMem[nbWritten] = 0;
			}
			return ptr;

		}

		public void Save(string filename, bool metaOnly = false)
		{
			ThrowIfNotInitialized();
			Native.gguf_write_to_file(this, filename, metaOnly);
		}

		public void AddTensor(SafeGGmlTensor tensor)
		{
			Native.gguf_add_tensor(this, tensor);
		}


		public string GetTensorName(int index)
		{
			return Native.gguf_get_tensor_name(this, index);
		}

		public ulong GetDataOffset()
		{
			return Native.gguf_get_data_offset(this);
		} 

		public ulong GetTensorOffset(int i)
		{
			return Native.gguf_get_tensor_offset(this, i);
		}



	}
}
