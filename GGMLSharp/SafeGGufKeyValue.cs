using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGufKeyValue 
	{
		private gguf_kv gguf_kv;

		public SafeGGufKeyValue()
		{

		}

		internal SafeGGufKeyValue(gguf_kv kv)
		{
			this.gguf_kv = kv;
		}

		public string Key => Marshal.PtrToStringAnsi(gguf_kv.key.data);
		public Structs.GGufType Type => (Structs.GGufType)gguf_kv.type;
		public object Value
		{
			get
			{
				switch (Type)
				{
					case Structs.GGufType. GGUF_TYPE_BOOL:
						return gguf_kv.value.bool_;
					case Structs.GGufType. GGUF_TYPE_COUNT:
						return gguf_kv.value.uint32;
					case Structs.GGufType. GGUF_TYPE_FLOAT32:
						return gguf_kv.value.float32;
					case Structs.GGufType. GGUF_TYPE_FLOAT64:
						return gguf_kv.value.float64;
					case Structs.GGufType. GGUF_TYPE_INT16:
						return gguf_kv.value.int16;
					case Structs.GGufType. GGUF_TYPE_INT32:
						return gguf_kv.value.int32;
					case Structs.GGufType. GGUF_TYPE_INT64:
						return gguf_kv.value.int64;
					case Structs.GGufType. GGUF_TYPE_INT8:
						return gguf_kv.value.int8;
					case Structs.GGufType. GGUF_TYPE_STRING:
						return Marshal.PtrToStringAnsi(gguf_kv.value.str.data);
					case Structs.GGufType. GGUF_TYPE_UINT16:
						return gguf_kv.value.uint16;
					case Structs.GGufType. GGUF_TYPE_UINT32:
						return gguf_kv.value.uint32;
					case Structs.GGufType. GGUF_TYPE_UINT64:
						return gguf_kv.value.uint64;
					case Structs.GGufType. GGUF_TYPE_UINT8:
						return gguf_kv.value.uint8;
					case Structs.GGufType. GGUF_TYPE_ARRAY:
						switch (gguf_kv.value._arr.type)
						{
							case gguf_type.GGUF_TYPE_BOOL:
								{
									bool[] arr = new bool[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<bool>(gguf_kv.value._arr.data + (int)(i * sizeof(bool)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_FLOAT32:
								{
									float[] arr = new float[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<float>(gguf_kv.value._arr.data + (int)(i * sizeof(float)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_FLOAT64:
								{
									double[] arr = new double[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<double>(gguf_kv.value._arr.data + (int)(i * sizeof(double)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_INT16:
								{
									Int16[] arr = new Int16[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<Int16>(gguf_kv.value._arr.data + (int)(i * sizeof(Int16)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_INT32:
								{
									Int32[] arr = new Int32[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<Int32>(gguf_kv.value._arr.data + (int)(i * sizeof(Int32)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_INT64:
								{
									Int64[] arr = new Int64[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<Int64>(gguf_kv.value._arr.data + (int)(i * sizeof(Int64)));
									}
									return arr;
								}

							case gguf_type.GGUF_TYPE_INT8:
								{
									SByte[] arr = new SByte[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<SByte>(gguf_kv.value._arr.data + (int)(i * sizeof(SByte)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_UINT16:
								{
									UInt16[] arr = new UInt16[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<UInt16>(gguf_kv.value._arr.data + (int)(i * sizeof(UInt16)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_UINT32:
								{
									UInt32[] arr = new UInt32[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<UInt32>(gguf_kv.value._arr.data + (int)(i * sizeof(UInt32)));
									}
									return arr;
								}

							case gguf_type.GGUF_TYPE_UINT64:
								{
									UInt64[] arr = new UInt64[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<UInt64>(gguf_kv.value._arr.data + (int)(i * sizeof(UInt64)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_UINT8:
								{
									byte[] arr = new byte[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										arr[i] = Marshal.PtrToStructure<byte>(gguf_kv.value._arr.data + (int)(i * sizeof(byte)));
									}
									return arr;
								}
							case gguf_type.GGUF_TYPE_STRING:
								{
									string[] arr = new string[gguf_kv.value._arr.n];
									for (ulong i = 0; i < gguf_kv.value._arr.n; i++)
									{
										byte[] bytes = new byte[4];
										gguf_str str = Marshal.PtrToStructure<gguf_str>(gguf_kv.value._arr.data + (int)i * sizeof(gguf_str));
										arr[i] = Marshal.PtrToStringAnsi(str.data);
									}
									return arr;
								}

							default:
								return null;
						}
					default:
						return null;
				}
			}
		}

	}
}
