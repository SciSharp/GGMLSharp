using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;
using ggml_backend_buffer_context_t = System.IntPtr;
using ggml_backend_graph_plan_t = System.IntPtr;
using ggml_backend_sched = System.IntPtr;
using ggml_fp16_t = System.UInt16;
using int16_t = System.Int16;
using int32_t = System.Int32;
using int64_t = System.Int64;
using int8_t = System.SByte;
using size_t = System.UInt64;
using uint16_t = System.UInt16;
using uint32_t = System.UInt32;
using uint64_t = System.UInt64;
using uint8_t = System.Byte;

namespace GGMLSharp
{
	internal unsafe class Native
	{
		const string DllName = "ggml";

		#region ggml.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ggml_status_to_string")]
		public extern static string ggml_status_to_string(ggml_status status);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_fp16_to_fp32(ggml_fp16_t x);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_fp16_t ggml_fp32_to_fp16(float x);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint16_t ggml_fp32_to_bf16(float x);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_bf16_to_fp32(uint16_t x);  // consider just doing << 16

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_bf16_to_fp32_row(uint16_t* x, float* y, int64_t n);

		public static float[] ggml_bf16_to_fp32_row(uint16_t[] x)
		{
			float[] y = new float[x.Length];
			ggml_bf16_to_fp32_row(Marshal.UnsafeAddrOfPinnedArrayElement(x, 0), Marshal.UnsafeAddrOfPinnedArrayElement(y, 0), x.Length);
			return y;
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		extern static void ggml_bf16_to_fp32_row(IntPtr x, IntPtr y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp32_to_bf16_row(float* x, uint16_t* y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp16_to_fp32_row(ggml_fp16_t* x, float* y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp32_to_fp16_row(float* x, ggml_fp16_t* y, int64_t n);

		public static ggml_fp16_t[] ggml_fp32_to_fp16_row(float[] x)
		{
			ggml_fp16_t[] y = new ggml_fp16_t[x.Length];
			ggml_fp32_to_fp16_row(Marshal.UnsafeAddrOfPinnedArrayElement(x, 0), Marshal.UnsafeAddrOfPinnedArrayElement(y, 0), x.Length);
			return y;
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		extern static void ggml_fp32_to_fp16_row(IntPtr x, IntPtr y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static void ggml_time_init(); // call this once at the beginning of the program
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static int64_t ggml_time_ms();
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_time_us();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_cycles();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_cycles_per_ms();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static void ggml_print_backtrace();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		// accepts a UTF-8 path, even on Windows
		public extern static IntPtr ggml_fopen(string fname, string mode);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_numa_init(ggml_numa_strategy numa); // call once for better performance on NUMA systems
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_numa(); // true if init detected that system has >1 NUMA node

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_print_object(SafeGGmlObject obj);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_print_objects(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_nelements(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_nrows(SafeGGmlTensor tensor);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_nbytes(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_nbytes_pad(SafeGGmlTensor tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_blck_size(Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_type_size(Structs.GGmlType type);             // size in bytes for all elements in a block

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_row_size(Structs.GGmlType type, int64_t ne); // size in bytes for all elements in a row

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_type_name(Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_op_name(ggml_op op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_op_symbol(ggml_op op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_unary_op_name(ggml_unary_op op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_op_desc(SafeGGmlTensor t); // unary or op name

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_element_size(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_quantized(Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		// TODO: temporary until model loading of ggml examples is refactored
		public extern static Structs.GGmlType ggml_ftype_to_ggml_type(ggml_ftype ftype);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_transposed(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_contiguous(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_permuted(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_empty(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_scalar(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_vector(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_matrix(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_3d(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_n_dims(SafeGGmlTensor tensor); // returns 1 for scalars

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_are_same_shape(SafeGGmlTensor t0, SafeGGmlTensor t1);

		/// <summary>
		/// use this to compute the memory overhead of a tensor
		/// </summary>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_tensor_overhead();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_validate_row_data(Structs.GGmlType type, IntPtr data, size_t nbytes);

		// main
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static IntPtr ggml_init(ggml_init_params @params);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_free(IntPtr ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_used_mem(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_set_scratch(SafeGGmlContext ctx, ggml_scratch scratch);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_get_no_alloc(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_no_alloc(SafeGGmlContext ctx, bool no_alloc);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_get_mem_buffer(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_get_mem_size(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_get_max_tensor_size(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_tensor(SafeGGmlContext ctx, Structs.GGmlType type, int n_dims, int64_t[] ne);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_tensor_1d(SafeGGmlContext ctx, Structs.GGmlType type, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_tensor_2d(SafeGGmlContext ctx, Structs.GGmlType type, int64_t ne0, int64_t ne1);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_tensor_3d(SafeGGmlContext ctx, Structs.GGmlType type, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_tensor_4d(SafeGGmlContext ctx, Structs.GGmlType type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static IntPtr ggml_new_tensor_4d(IntPtr ctx, Structs.GGmlType type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_i32(SafeGGmlContext ctx, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_new_f32(SafeGGmlContext ctx, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_dup_tensor(SafeGGmlContext ctx, SafeGGmlTensor src);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_view_tensor(SafeGGmlContext ctx, SafeGGmlTensor src);

		// Context tensor  eration and lookup
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_first_tensor(SafeGGmlContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_next_tensor(SafeGGmlContext ctx, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_tensor(SafeGGmlContext ctx, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_zero(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_i32(SafeGGmlTensor tensor, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_f32(SafeGGmlTensor tensor, float value);

		// Converts a flat index into coordinates
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_unravel_index(SafeGGmlTensor tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t ggml_get_i32_1d(SafeGGmlTensor tensor, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_i32_1d(SafeGGmlTensor tensor, int i, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t ggml_get_i32_nd(SafeGGmlTensor tensor, int i0, int i1, int i2, int i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_i32_nd(SafeGGmlTensor tensor, int i0, int i1, int i2, int i3, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_get_f32_1d(SafeGGmlTensor tensor, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_f32_1d(SafeGGmlTensor tensor, int i, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_get_f32_nd(SafeGGmlTensor tensor, int i0, int i1, int i2, int i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_f32_nd(SafeGGmlTensor tensor, int i0, int i1, int i2, int i3, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_get_data(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float* ggml_get_data_f32(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_unary_op ggml_get_unary_op(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_get_name(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_name(SafeGGmlTensor tensor, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_format_name(SafeGGmlTensor tensor, string fmt);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_dup(SafeGGmlContext ctx, SafeGGmlTensor a);

		// in-place, returns view(a)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_dup_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add_cast(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add1(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add1_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		// dst = a
		// view(dst, nb1, nb2, nb3, offset) += b
		// return dst
		public extern static SafeGGmlTensor ggml_acc(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_acc_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sub(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sub_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mul(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mul_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_div(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_div_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sqr(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sqr_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sqrt(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sqrt_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_log(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_log_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);


		/// <summary>
		/// return scalar
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sum(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sum_rows(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// mean along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mean(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// argmax along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_argmax(SafeGGmlContext ctx, SafeGGmlTensor a);


		/// <summary>
		/// if a is the same shape as b, and a is not parameter, return a otherwise, return a new tensor: repeat(a) to fit in b
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_repeat(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// sums repetitions in a into shape of b
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_repeat_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// concat a and b on dim 2 used in stable-diffusion
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_concat(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_abs(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_abs_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sgn(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sgn_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_neg(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_neg_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_step(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_step_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_tanh(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_tanh_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_elu(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_elu_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_relu(SafeGGmlContext ctx, SafeGGmlTensor a);

		// contains in ggml but not in llama.cpp
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sigmoid(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_leaky_relu(SafeGGmlContext ctx, SafeGGmlTensor a, float negative_slope, bool inplace);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_relu_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		// contains in ggml but not in llama.cpp
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_sigmoid_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_gelu(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_gelu_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_gelu_quick(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_gelu_quick_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_silu(SafeGGmlContext ctx, SafeGGmlTensor a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_silu_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		// a - x
		// b - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_silu_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// hardswish(x) = x * relu6(x + 3) / 6
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_hardswish(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// hardsigmoid(x) = relu6(x + 3) / 6
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_hardsigmoid(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// normalize along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="eps"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_norm(SafeGGmlContext ctx, SafeGGmlTensor a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_norm_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rms_norm(SafeGGmlContext ctx, SafeGGmlTensor a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rms_norm_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, float eps);

		/// <summary>
		/// group normalize along ne0*ne1*n_groups
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_groups"></param>
		/// <returns></returns>
		// used in stable-diffusion
		// TODO: Eps is hardcoded to 1e-6 for now
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_group_norm(SafeGGmlContext ctx, SafeGGmlTensor a, int n_groups);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_group_norm_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, int n_groups);

		// a - x
		// b - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rms_norm_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, float eps);

		/// <summary>
		/// result is n columns, NumberOfCorrections rows => [ne03 * x, ne02 * y, NumberOfCorrections, n]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a">k columns, n rows => [ne03, ne02, n, k]</param>
		/// <param name="b">k columns, NumberOfCorrections rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, NumberOfCorrections, k]</param>
		/// <returns></returns>
		// 
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mul_mat(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// change the precision of a matrix multiplication set to GGML_PREC_F32 for higher precision (useful for phi-2)
		/// </summary>
		/// <param name="a"></param>
		/// <param name="prec"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_mul_mat_set_prec(SafeGGmlTensor a, ggml_prec prec);

		/// <summary>
		/// this func contains in ggml but not in llama.cpp. indirect matrix multiplication ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b) 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="as"></param>
		/// <param name="ids"></param>
		/// <param name="id"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mul_mat_id(SafeGGmlContext ctx, SafeGGmlTensor @as, SafeGGmlTensor ids, int id, SafeGGmlTensor b);


		/// <summary>
		/// this func contains in llama.cpp but not in ggml. indirect matrix multiplication ggml_mul_mat_id(ctx, as, ids, id, b) ~= ggml_mul_mat(as[ids[id]], b) 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="as"></param>
		/// <param name="ids"></param>
		/// <param name="id"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_mul_mat_id(SafeGGmlContext ctx, SafeGGmlTensor @as, SafeGGmlTensor ids, SafeGGmlTensor b);

		/// <summary>
		/// result is NumberOfCorrections columns, p rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a">NumberOfCorrections columns, n rows,</param>
		/// <param name="b">p columns, n rows,</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_out_prod(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// operations on tensors without backpropagation
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="s"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_scale(SafeGGmlContext ctx, SafeGGmlTensor a, float s);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="s"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_scale_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, float s);

		/// <summary>
		/// b -> view(a,offset,nb1,nb2,3), return modified a
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="nb1"></param>
		/// <param name="nb2"></param>
		/// <param name="nb3"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		/// <summary>
		/// b -> view(a,offset,nb1,nb2,3), return view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="nb1"></param>
		/// <param name="nb2"></param>
		/// <param name="nb3"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_1d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_1d_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t offset);

		// b -> view(a,offset,nb1,nb2,3), return modified a
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_2d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t offset);

		/// <summary>
		/// b -> view(a,offset,nb1,nb2,3), return view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="nb1"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_set_2d_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, size_t nb1, size_t offset);

		/// <summary>
		/// a -> b, return view(b)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cpy(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cast(SafeGGmlContext ctx, SafeGGmlTensor a, Structs.GGmlType type);

		/// <summary>
		/// make contiguous
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cont(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// make contiguous, with new shape
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cont_1d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cont_2d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cont_3d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cont_4d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		/// <summary>
		/// return view(a), b specifies the new shape
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		// TODO: when we start computing gradient, make a copy instead of view
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_reshape(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// return view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <returns></returns>
		// TODO: when we start computing gradient, make a copy instead of view
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_reshape_1d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_reshape_2d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1);

		/// <summary>
		/// return view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="ne1"></param>
		/// <param name="ne2"></param>
		/// <returns></returns>
		// TODO: when we start computing gradient, make a copy instead of view
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_reshape_3d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_reshape_4d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		/// <summary>
		/// offset in bytes
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_view_1d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, size_t offset);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="ne1"></param>
		/// <param name="nb1">row stride in bytes</param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_view_2d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="ne1"></param>
		/// <param name="ne2"></param>
		/// <param name="nb1">row stride in bytes</param>
		/// <param name="nb2">slice stride in bytes</param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_view_3d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="ne1"></param>
		/// <param name="ne2"></param>
		/// <param name="ne3"></param>
		/// <param name="nb1">row stride in bytes</param>
		/// <param name="nb2">slice stride in bytes</param>
		/// <param name="nb3"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_view_4d(SafeGGmlContext ctx, SafeGGmlTensor a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_permute(SafeGGmlContext ctx, SafeGGmlTensor a, int axis0, int axis1, int axis2, int axis3);

		/// <summary>
		/// alias for Permute(ctx, a, 1, 0, 2, 3)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_transpose(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// supports 3D: a->ne[2] == b->ne[1]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_rows(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_rows_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, SafeGGmlTensor c);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_diag(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// set elements above the diagonal to -INF
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_diag_mask_inf(SafeGGmlContext ctx, SafeGGmlTensor a, int n_past);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_diag_mask_inf_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, int n_past);

		/// <summary>
		/// set elements above the diagonal to 0
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_diag_mask_zero(SafeGGmlContext ctx, SafeGGmlTensor a, int n_past);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_diag_mask_zero_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, int n_past);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_soft_max(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_soft_max_inplace(SafeGGmlContext ctx, SafeGGmlTensor a);

		/// <summary>
		/// fused soft_max(a*scale + mask + pos[i]*(ALiBi slope))
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="mask"> </param>
		/// <param name="pos">required when max_bias > 0.0f</param>
		/// <param name="scale"></param>
		/// <param name="max_bias">0.0f for no ALiBi</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_soft_max_ext(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor mask, float scale, float max_bias);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_soft_max_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_soft_max_back_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		// rotary position embedding
		// if mode & 1 == 1, skip n_past elements (DEPRECATED)
		// if mode & 2 == 1, GPT-NeoX style
		// if mode & 4 == 1, ChatGLM style
		//
		// b is an int32 vector with size a->ne[2], it contains the positions
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, int mode, int n_ctx);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="n_dims"></param>
		/// <param name="mode"></param>
		/// <param name="n_ctx"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, int mode, int n_ctx);

		/// <summary>
		/// custom RoPE
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="n_dims"></param>
		/// <param name="mode"></param>
		/// <param name="n_ctx"></param>
		/// <param name="n_orig_ctx"></param>
		/// <param name="freq_base"></param>
		/// <param name="freq_scale"></param>
		/// <param name="ext_factor"></param>
		/// <param name="attn_factor"></param>
		/// <param name="beta_fast"></param>
		/// <param name="beta_slow"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope_custom(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="n_dims"></param>
		/// <param name="mode"></param>
		/// <param name="n_ctx"></param>
		/// <param name="n_orig_ctx"></param>
		/// <param name="freq_base"></param>
		/// <param name="freq_scale"></param>
		/// <param name="ext_factor"></param>
		/// <param name="attn_factor"></param>
		/// <param name="beta_fast"></param>
		/// <param name="beta_slow"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope_custom_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

		/// <summary>
		/// compute correction dims for YaRN RoPE scaling
		/// </summary>
		/// <param name="n_dims"></param>
		/// <param name="n_orig_ctx"></param>
		/// <param name="freq_base"></param>
		/// <param name="beta_fast"></param>
		/// <param name="beta_slow"></param>
		/// <param name="dims"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_rope_yarn_corr_dims(int n_dims, int n_orig_ctx, float freq_base, float beta_fast, float beta_slow, float[] dims);

		/// <summary>
		/// xPos RoPE, in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="n_dims"></param>
		/// <param name="base"></param>
		/// <param name="down"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope_xpos_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, float @base, bool down);

		// rotary position embedding backward, i.e compute dx from dy
		// a - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_rope_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float xpos_base, bool xpos_down);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_clamp(SafeGGmlContext ctx, SafeGGmlTensor a, float min, float max);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_im2col(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D, Structs.GGmlType dst_type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_depthwise_2d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s0, int s1, int p0, int p1, int d0, int d1);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="s0">stride</param>
		/// <param name="p0">padding</param>
		/// <param name="d0">dilation</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_1d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s0, int p0, int d0);

		// conv_1d with padding = half
		// alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_1d_ph(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s, int d);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static SafeGGmlTensor ggml_conv_transpose_1d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s0, int p0, int d0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_2d(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int s0, int s1, int p0, int p1, int d0, int d1);


		// kernel size is a->ne[0] x a->ne[1]
		// stride is equal to kernel size
		// padding is zero
		// example:
		// a:     16   16    3  768
		// b:   1024 1024    3    1
		// res:   64   64  768    1
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_2d_sk_p0(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		// kernel size is a->ne[0] x a->ne[1]
		// stride is 1
		// padding is half
		// example:
		// a:      3    3    256  256
		// b:     64   64    256    1
		// res:   64   64    256    1
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_2d_s1_ph(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_conv_transpose_2d_p0(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, int stride);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="op"></param>
		/// <param name="k0">kernel size</param>
		/// <param name="s0">stride</param>
		/// <param name="p0">padding</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_pool_1d(SafeGGmlContext ctx, SafeGGmlTensor a, ggml_op_pool op, int k0, int s0, int p0);

		// the result will have 2*p0 padding for the first dimension
		// and 2*p1 padding for the second dimension
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_pool_2d(SafeGGmlContext ctx, SafeGGmlTensor a, ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

		// nearest interpolate
		// used in stable-diffusion
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_upscale(SafeGGmlContext ctx, SafeGGmlTensor a, int scale_factor);

		// pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_pad(SafeGGmlContext ctx, SafeGGmlTensor a, int p0, int p1, int p2, int p3);

		// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
		// timesteps: [N,]
		// return: [N, dim]
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_timestep_embedding(SafeGGmlContext ctx, SafeGGmlTensor timesteps, int dim, int max_period);

		// sort rows

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_argsort(SafeGGmlContext ctx, SafeGGmlTensor a, ggml_sort_order order);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_arange(SafeGGmlContext ctx, float start, float stop, float step);

		// top k elements per row
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_top_k(SafeGGmlContext ctx, SafeGGmlTensor a, int k);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_flash_attn(SafeGGmlContext ctx, SafeGGmlTensor q, SafeGGmlTensor k, SafeGGmlTensor v, bool masked);

		/// <summary>
		/// 
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="q">[n_embd, n_batch, n_head, 1]</param>
		/// <param name="k">[n_embd, n_kv, n_head_kv, 1]</param>
		/// <param name="v"></param>
		/// <param name="mask">[n_kv, n_batch_pad, 1, 1] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!</param>
		/// <param name="scale">[n_embd, n_head, n_batch, 1] !! permuted !!</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_flash_attn_ext(SafeGGmlContext ctx, SafeGGmlTensor q, SafeGGmlTensor k, SafeGGmlTensor v, SafeGGmlTensor mask, float scale, float max_bias);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_flash_attn_ext_set_prec(SafeGGmlTensor a, ggml_prec prec);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_flash_attn_back(SafeGGmlContext ctx, SafeGGmlTensor q, SafeGGmlTensor k, SafeGGmlTensor v, SafeGGmlTensor d, bool masked);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_flash_ff(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b0, SafeGGmlTensor b1, SafeGGmlTensor c0, SafeGGmlTensor c1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_ssm_conv(SafeGGmlContext ctx, SafeGGmlTensor s, SafeGGmlTensor x, SafeGGmlTensor c, SafeGGmlTensor sq);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_ssm_scan(SafeGGmlContext ctx, SafeGGmlTensor s, SafeGGmlTensor x, SafeGGmlTensor dt, SafeGGmlTensor A, SafeGGmlTensor B, SafeGGmlTensor C, SafeGGmlTensor sq);

		// partition into non-overlapping windows with padding if needed
		// example:
		// a:   768   64   64    1
		// w:    14
		// res: 768   14   14    25
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_win_part(SafeGGmlContext ctx, SafeGGmlTensor a, int w);

		/// <summary>
		/// reverse of WinPart used in sam
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="w0"></param>
		/// <param name="h0"></param>
		/// <param name="w"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_win_unpart(SafeGGmlContext ctx, SafeGGmlTensor a, int w0, int h0, int w);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_unary(SafeGGmlContext ctx, SafeGGmlTensor a, ggml_unary_op op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_unary_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, ggml_unary_op op);

		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_get_rel_pos(SafeGGmlContext ctx, SafeGGmlTensor a, int qh, int kh);

		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add_rel_pos(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor pw, SafeGGmlTensor ph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_add_rel_pos_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor pw, SafeGGmlTensor ph);



		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom1(SafeGGmlContext ctx, SafeGGmlTensor a, [MarshalAs(UnmanagedType.FunctionPtr)] Structs.Custom1OpDelegate fun, int n_tasks, IntPtr userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom1_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, [MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(Structs.Custom1OpDelegate))] Structs.Custom1OpDelegate fun, int n_tasks, IntPtr userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom2(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, [MarshalAs(UnmanagedType.FunctionPtr)] Structs.Custom2OpDelegate fun, int n_tasks, IntPtr userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom2_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, [MarshalAs(UnmanagedType.FunctionPtr)] Structs.Custom2OpDelegate fun, int n_tasks, IntPtr userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom3(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, SafeGGmlTensor c, [MarshalAs(UnmanagedType.FunctionPtr)] Structs.Custom3OpDelegate fun, int n_tasks, IntPtr userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_map_custom3_inplace(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, SafeGGmlTensor c, [MarshalAs(UnmanagedType.FunctionPtr)] Structs.Custom3OpDelegate fun, int n_tasks, IntPtr userdata);

		// loss function

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cross_entropy_loss(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_cross_entropy_loss_back(SafeGGmlContext ctx, SafeGGmlTensor a, SafeGGmlTensor b, SafeGGmlTensor c);

		//
		// automatic differentiation
		//

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_param(SafeGGmlContext ctx, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_forward_expand(SafeGGmlGraph cgraph, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_backward_expand(SafeGGmlContext ctx, SafeGGmlGraph gf, SafeGGmlGraph gb, bool keep);

		// graph allocation in a context
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlGraph ggml_new_graph(SafeGGmlContext ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlGraph ggml_new_graph_custom(SafeGGmlContext ctx, size_t size, bool grads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlGraph ggml_graph_dup(SafeGGmlContext ctx, SafeGGmlGraph cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph ggml_graph_view(SafeGGmlGraph cgraph, int i0, int i1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_cpy(SafeGGmlGraph src, SafeGGmlGraph dst);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_reset(SafeGGmlGraph cgraph);  // zero grads

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_clear(SafeGGmlGraph cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_graph_overhead();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_graph_overhead_custom(size_t size, bool grads);

		// ggml_graph_plan() has to be called before ggml_graph_compute()
		// when plan.work_size > 0, caller must allocate memory for plan.work_data
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cplan ggml_graph_plan(SafeGGmlGraph cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_graph_compute(SafeGGmlGraph cgraph, ggml_cplan* cplan);
		// same as ggml_graph_compute() but the work data is allocated as a part of the context
		// note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_graph_compute_with_ctx(SafeGGmlContext ctx, SafeGGmlGraph cgraph, int n_threads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlTensor ggml_graph_get_tensor(SafeGGmlGraph cgraph, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_export(SafeGGmlGraph cgraph, string fname);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlGraph ggml_graph_import(string fname, ggml_context** ctx_data, ggml_context** ctx_eval);

		// print info and performance information for the graph
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_print(SafeGGmlGraph cgraph);

		// dump the graph into a file using the dot format
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_dump_dot(SafeGGmlGraph gb, SafeGGmlGraph gf, string filename);

		// build gradient checkpointing backward graph gb for gf using provided checkpoints
		// gb_tmp will contain original backward graph with rewritten backward process nodes,
		// but without the second forward pass nodes.
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_backward_gradient_checkpointing(SafeGGmlContext ctx, SafeGGmlGraph gf, SafeGGmlGraph gb, SafeGGmlGraph gb_tmp, ggml_tensor** checkpoints, int n_checkpoints);

		public delegate void ggml_opt_callback(void* data, int accum_step, float* sched, bool* cancel);
		public delegate void ggml_log_callback(ggml_log_level level, string text, void* user_data);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.OptimizerParameters ggml_opt_default_params(Structs.OptimizerType type);

		// optimize the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.OptimizationResult ggml_opt(SafeGGmlContext ctx, Structs.OptimizerParameters @params, SafeGGmlTensor f);

		// optimize the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.OptimizationResult ggml_opt(IntPtr ctx, Structs.OptimizerParameters @params, SafeGGmlTensor f);

		// initialize optimizer context
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_opt_init(SafeGGmlContext ctx, ggml_opt_context* opt, ggml_opt_params @params, int64_t nx);

		// continue optimizing the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_result ggml_opt_resume(SafeGGmlContext ctx, ggml_opt_context* opt, SafeGGmlTensor f);

		// continue optimizing the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_result ggml_opt_resume_g(SafeGGmlContext ctx, ggml_opt_context* opt, SafeGGmlTensor f, SafeGGmlGraph gf, SafeGGmlGraph gb, ggml_opt_callback callback, void* callback_data);

		//
		// tensor flags
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_input(SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_output(SafeGGmlTensor tensor);

		//
		// quantization
		//

		// - ggml_quantize_init can be called multiple times with the same Type
		//   it will only initialize the quantization tables for the first call or after ggml_quantize_free
		//   automatically called by ggml_quantize_chunk for convenience
		//
		// - ggml_quantize_free will free any memory allocated by ggml_quantize_init
		//   call this at the end of the program to avoid memory leaks
		//
		// note: these are thread-safe
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_quantize_init(Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_quantize_free();

		// some quantization Type cannot be used without an importance matrix
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_quantize_requires_imatrix(Structs.GGmlType type);

		// calls ggml_quantize_init internally (i.e. can allocate memory)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_quantize_chunk(Structs.GGmlType type, float* src, void* dst, int64_t start, int64_t nrows, int64_t n_per_row, float* imatrix);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGufContext gguf_init_empty();

		public static SafeGGufContext gguf_init_from_file(string fname, SafeGGmlContext ggmlContext, bool noAlloc)
		{
			ggml_context* context = (ggml_context*)Marshal.AllocHGlobal(sizeof(ggml_context));
			gguf_init_params init_params = new gguf_init_params
			{
				no_alloc = noAlloc,
				ctx = &context,
			};
			SafeGGufContext gguf_ctx = gguf_init_from_file_native(fname, init_params);
			ggmlContext.SetContext(context);
			return gguf_ctx;
		}

		[DllImport(DllName, EntryPoint = "gguf_init_from_file", CallingConvention = CallingConvention.Cdecl)]

		public extern static SafeGGufContext gguf_init_from_file_native(string fname, gguf_init_params @params);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static gguf_context* gguf_init_from_file(string fname, gguf_init_params @params);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_free(IntPtr ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gguf_type_name")]
		public extern static string gguf_type_name(Structs.GGufType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_version(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_alignment(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_data_offset(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_data(SafeGGufContext ctx);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_n_kv(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool gguf_find_key(SafeGGufContext ctx, string key);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string gguf_get_key(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.GGufType gguf_get_kv_type(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.GGufType gguf_get_arr_type(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_arr_data(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string gguf_get_arr_str(SafeGGufContext ctx, int key_id, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_arr_n(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint8_t gguf_get_val_u8(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int8_t gguf_get_val_i8(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint16_t gguf_get_val_u16(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int16_t gguf_get_val_i16(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint32_t gguf_get_val_u32(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t gguf_get_val_i32(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float gguf_get_val_f32(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint64_t gguf_get_val_u64(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t gguf_get_val_i64(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static double gguf_get_val_f64(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool gguf_get_val_bool(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string gguf_get_val_str(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_val_data(SafeGGufContext ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_n_tensors(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_find_tensor(SafeGGufContext ctx, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_tensor_offset(SafeGGufContext ctx, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string gguf_get_tensor_name(SafeGGufContext ctx, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static Structs.GGmlType gguf_get_tensor_type(SafeGGufContext ctx, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_remove_key(SafeGGufContext ctx, string key);

		// returns the index
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_or_add_key(SafeGGufContext ctx, string key);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u8(SafeGGufContext ctx, string key, uint8_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i8(SafeGGufContext ctx, string key, int8_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u16(SafeGGufContext ctx, string key, uint16_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i16(SafeGGufContext ctx, string key, int16_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u32(SafeGGufContext ctx, string key, uint32_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i32(SafeGGufContext ctx, string key, int32_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_f32(SafeGGufContext ctx, string key, float val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u64(SafeGGufContext ctx, string key, uint64_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i64(SafeGGufContext ctx, string key, int64_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_f64(SafeGGufContext ctx, string key, double val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_bool(SafeGGufContext ctx, string key, bool val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_str(SafeGGufContext ctx, string key, string val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_arr_data(SafeGGufContext ctx, string key, Structs.GGufType type, IntPtr data, int n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		//public extern static void gguf_set_arr_str(SafeGGufContext ctx, string key, IntPtr data, int n);
		public extern static void gguf_set_arr_str(SafeGGufContext ctx, string key, string[] data, int n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		//public extern static void gguf_set_arr_str(SafeGGufContext ctx, string key, IntPtr data, int n);
		public extern static void gguf_set_arr_str(SafeGGufContext ctx, string key, IntPtr[] data, int n);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void gguf_set_kv(SafeGGufContext ctx, SafeGGufContext src);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_add_tensor(SafeGGufContext ctx, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_tensor_type(SafeGGufContext ctx, string name, Structs.GGmlType type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_tensor_data(SafeGGufContext ctx, string name, IntPtr data, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_write_to_file(SafeGGufContext ctx, string fname, bool only_meta);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_meta_size(SafeGGufContext ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_get_meta_data(SafeGGufContext ctx, IntPtr data);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx_vnni();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx2();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx512();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx512_vbmi();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx512_vnni();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_avx512_bf16();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_fma();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_neon();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_arm_fma();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_metal();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_f16c();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_fp16_va();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_wasm_simd();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_blas();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_cuda();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_clblast();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_vulkan();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_kompute();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_sycl();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_gpublas();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_sse3();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_ssse3();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_vsx();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_cpu_has_matmul_int8();

		public static int GGML_PAD(int x, int n)
		{
			return (x + n - 1) & ~(n - 1);
		}

		#endregion


		#region ggml-backend-impl.h

		public delegate SafeGGmlBackend ggml_backend_init_fn(string @params, IntPtr user_data);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_buffer_init(SafeGGmlBackendBufferType buft, ggml_backend_buffer_i iface, ggml_backend_buffer_context_t context, size_t size);

		/// <summary>
		/// do not use directly, use ggml_backend_tensor_copy instead
		/// </summary>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_copy_tensor(SafeGGmlTensor src, SafeGGmlTensor dst);

		/// <summary>
		/// buffer that contains a collection of buffers
		/// </summary>
		/// <param name="buffers"></param>
		/// <param name="n_buffers"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer** buffers, size_t n_buffers);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_is_multi_buffer(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_multi_buffer_set_usage(SafeGGmlBackendBuffer buffer, ggml_backend_buffer_usage usage);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_register(string name, ggml_backend_init_fn init_fn, SafeGGmlBackendBufferType default_buffer_type, IntPtr user_data);

		#endregion


		#region ggml-backend.h

		// buffer Type
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_backend_buft_name(SafeGGmlBackendBufferType buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_buft_alloc_buffer(SafeGGmlBackendBufferType buft, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_alignment(SafeGGmlBackendBufferType buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_max_size(SafeGGmlBackendBufferType buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_alloc_size(SafeGGmlBackendBufferType buft, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buft_supports_backend(SafeGGmlBackendBufferType buft, SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buft_is_host(SafeGGmlBackendBufferType buft);


		/// <summary>
		/// Copy a graph to a different backend
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_graph_copy ggml_backend_graph_copy(SafeGGmlBackend backend, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_graph_copy_free(ggml_backend_graph_copy copy);


		[DllImport(DllName, EntryPoint = "ggml_backend_buffer_name", CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_backend_buffer_name(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_free(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_backend_buffer_get_base(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_size(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_init_tensor(SafeGGmlBackendBuffer buffer, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_alignment(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_max_size(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_alloc_size(SafeGGmlBackendBuffer buffer, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_clear(SafeGGmlBackendBuffer buffer, uint8_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_is_host(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_set_usage(SafeGGmlBackendBuffer buffer, ggml_backend_buffer_usage usage);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBufferType ggml_backend_buffer_get_type(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_reset(SafeGGmlBackendBuffer buffer);

		//
		// Backend
		//

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_guid_t ggml_backend_guid(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_backend_name(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_free(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBufferType ggml_backend_get_default_buffer_type(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_alloc_buffer(SafeGGmlBackend backend, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_get_alignment(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_get_max_size(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_set_async(SafeGGmlBackend backend, SafeGGmlTensor tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_get_async(SafeGGmlBackend backend, SafeGGmlTensor tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_set(SafeGGmlTensor tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_get(SafeGGmlTensor tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_synchronize(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_graph_plan_t ggml_backend_graph_plan_create(SafeGGmlBackend backend, SafeGGmlGraph cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_graph_plan_free(SafeGGmlBackend backend, ggml_backend_graph_plan_t plan);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_plan_compute(SafeGGmlBackend backend, ggml_backend_graph_plan_t plan);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_compute(SafeGGmlBackend backend, SafeGGmlGraph cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_compute_async(SafeGGmlBackend backend, SafeGGmlGraph cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_supports_op(SafeGGmlBackend backend, SafeGGmlTensor op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_offload_op(SafeGGmlBackend backend, SafeGGmlTensor op);

		/// <summary>
		/// tensor copy between different backends
		/// </summary>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_copy(SafeGGmlTensor src, SafeGGmlTensor dst);

		// asynchronous copy
		// the copy is performed after all the currently queued operations in backend_src
		// backend_dst will wait for the copy to complete before performing other operations
		// automatic fallback to sync copy if async is not supported
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_copy_async(SafeGGmlBackend backend_src, SafeGGmlBackend backend_dst, SafeGGmlTensor src, SafeGGmlTensor dst);

		// events
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_event* ggml_backend_event_new(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_free(ggml_backend_event* @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_record(ggml_backend_event* @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_synchronize(ggml_backend_event* @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_wait(SafeGGmlBackend backend, ggml_backend_event* @event); // wait async on event

		//
		// CPU backend
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackend ggml_backend_cpu_init();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_is_cpu(SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_cpu_set_n_threads(SafeGGmlBackend backend_cpu, int n_threads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_cpu_set_abort_callback(SafeGGmlBackend backend_cpu, ggml_abort_callback abort_callback, void* abort_callback_data);

		/// <summary>
		/// Create a backend buffer from an existing pointer
		/// </summary>
		/// <param name="ptr"></param>
		/// <param name="size"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_cpu_buffer_from_ptr(IntPtr ptr, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBufferType ggml_backend_cpu_buffer_type();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBufferType ggml_backend_cpu_hbm_buffer_type();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_reg_get_count();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_reg_find_by_name(string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackend ggml_backend_reg_init_backend_from_str(string backend_str); // str is name[:params]

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static string ggml_backend_reg_get_name(size_t i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackend ggml_backend_reg_init_backend(size_t i, string @params); // params is backend-specific

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBufferType ggml_backend_reg_get_default_buffer_type(size_t i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_reg_alloc_buffer(size_t i, size_t size);

		/// <summary>
		/// Initialize a backend scheduler
		/// </summary>
		/// <param name="backends"></param>
		/// <param name="bufts"></param>
		/// <param name="n_backends"></param>
		/// <param name="graph_size"></param>
		/// <param name="parallel"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_sched* ggml_backend_sched_new(SafeGGmlBackend* backends, ggml_backend_buffer_type** bufts, int n_backends, size_t graph_size, bool parallel);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_free(ggml_backend_sched* sched);

		/// <summary>
		/// Initialize backend buffers from a measure graph
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="measure_graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_sched_reserve(ggml_backend_sched* sched, SafeGGmlGraph measure_graph);

		/// <summary>
		/// Get the number of splits of the last graph
		/// </summary>
		/// <param name="sched"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_backend_sched_get_n_splits(ggml_backend_sched* sched);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_backend_sched_get_n_copies(ggml_backend_sched* sched);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched* sched, SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_set_tensor_backend(ggml_backend_sched* sched, SafeGGmlTensor node, SafeGGmlBackend backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackend ggml_backend_sched_get_tensor_backend(ggml_backend_sched* sched, SafeGGmlTensor node);

		/// <summary>
		/// Allocate and compute graph on the backend scheduler
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_sched_alloc_graph(ggml_backend_sched* sched, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched* sched, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched* sched, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_synchronize(ggml_backend_sched* sched);

		/// <summary>
		/// Reset all assignments and allocators - must be called before changing the node backends
		/// </summary>
		/// <param name="sched"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_reset(ggml_backend_sched* sched);

		/// <summary>
		/// Set a callback to be called for each resulting node during graph compute
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="callback"></param>
		/// <param name="user_data"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_set_eval_callback(ggml_backend_sched* sched, ggml_backend_sched_eval_callback callback, void* user_data);

		/// <summary>
		/// Compare the output of two backends
		/// </summary>
		/// <param name="backend1"></param>
		/// <param name="backend2"></param>
		/// <param name="graph"></param>
		/// <param name="callback"></param>
		/// <param name="user_data"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_compare_graph_backend(SafeGGmlBackend backend1, SafeGGmlBackend backend2, SafeGGmlGraph graph, ggml_backend_eval_callback callback, void* user_data);

		/// <summary>
		/// Tensor initialization
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="tensor"></param>
		/// <param name="addr"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_alloc(SafeGGmlBackendBuffer buffer, SafeGGmlTensor tensor, void* addr);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_view_init(SafeGGmlBackendBuffer buffer, SafeGGmlTensor tensor);



		#endregion


		#region ggml_alloc.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tallocr ggml_tallocr_new(SafeGGmlBackendBuffer buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_tallocr_alloc(ggml_tallocr* talloc, SafeGGmlTensor tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_gallocr* ggml_gallocr_new(SafeGGmlBackendBufferType buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_gallocr* ggml_gallocr_new_n(ggml_backend_buffer_type** bufts, int n_bufs);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_gallocr_free(IntPtr galloc);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_reserve(SafeGGmlGraphAllocr galloc, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_reserve_n(SafeGGmlGraphAllocr galloc, SafeGGmlGraph graph, int* node_buffer_ids, int* leaf_buffer_ids);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_alloc_graph(SafeGGmlGraphAllocr galloc, SafeGGmlGraph graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_gallocr_get_buffer_size(SafeGGmlGraphAllocr galloc, int buffer_id);

		/// <summary>
		/// Create a buffer and allocate all the tensors in a ggml_context
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="buft"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_alloc_ctx_tensors_from_buft(SafeGGmlContext ctx, SafeGGmlBackendBufferType buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackendBuffer ggml_backend_alloc_ctx_tensors(SafeGGmlContext ctx, SafeGGmlBackend backend);


		#endregion

		#region ggml-impl.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_hash_set ggml_hash_set_new(size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_hash_contains(ggml_hash_set hash_set, SafeGGmlTensor key);

		// returns GGML_HASHTABLE_FULL if table is full, otherwise the current index of the key or where it should be inserted
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_find(ggml_hash_set hash_set, SafeGGmlTensor key);

		// returns GGML_HASHTABLE_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_insert(ggml_hash_set hash_set, SafeGGmlTensor key);

		// return index, asserts if table is full
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_find_or_insert(ggml_hash_set hash_set, SafeGGmlTensor key);


		#endregion

		#region ggml-cuda.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static SafeGGmlBackend ggml_backend_cuda_init(int device);

		#endregion

	}
}

