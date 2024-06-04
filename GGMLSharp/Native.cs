using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace GGMLSharp
{
	public unsafe class Native
	{

		const string DllName = "ggml";

		#region ggml.h

		public static string? ggml_status_to_string(ggml_status status)
		{
			return Marshal.PtrToStringUTF8(ggml_status_to_string_native(status));

			//[LibraryImport(DllName,StringMarshalling = StringMarshalling.Utf8,SetLastError = true)]
			[DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "ggml_status_to_string")]
			extern static IntPtr ggml_status_to_string_native(ggml_status status);
		}


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

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp32_to_bf16_row(float* x, uint16_t* y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp16_to_fp32_row(ggml_fp16_t* x, float* y, int64_t n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_fp32_to_fp16_row(float* x, ggml_fp16_t* y, int64_t n);

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
		public extern static void ggml_print_object(ggml_object* obj);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_print_objects(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_nelements(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t ggml_nrows(ggml_tensor* tensor);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_nbytes(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_nbytes_pad(ggml_tensor* tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_blck_size(ggml_type type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_type_size(ggml_type type);             // size in bytes for all elements in a block

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_row_size(ggml_type type, int64_t ne); // size in bytes for all elements in a row

		public static string? ggml_type_name(ggml_type type)
		{
			return Marshal.PtrToStringAnsi(ggml_type_name_native(type));
			[DllImport(DllName, EntryPoint = "ggml_type_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_type_name_native(ggml_type type);
		}

		public static string? ggml_op_name(ggml_op op)
		{
			return Marshal.PtrToStringAnsi(ggml_op_name_native(op));
			[DllImport(DllName, EntryPoint = "ggml_op_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_op_name_native(ggml_op op);
		}

		public static string? ggml_op_symbol(ggml_op op)
		{
			return Marshal.PtrToStringAnsi(ggml_op_symbol_native(op));
			[DllImport(DllName, EntryPoint = "ggml_op_symbol", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_op_symbol_native(ggml_op op);
		}

		public static string? ggml_unary_op_name(ggml_unary_op op)
		{
			return Marshal.PtrToStringAnsi(ggml_unary_op_name_native(op));
			[DllImport(DllName, EntryPoint = "ggml_unary_op_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_unary_op_name_native(ggml_unary_op op);
		}

		public static string? ggml_op_desc(ggml_tensor* t)
		{
			return Marshal.PtrToStringAnsi(ggml_op_desc_native(t));
			[DllImport(DllName, EntryPoint = "ggml_op_desc", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_op_desc_native(ggml_tensor* t); // unary or op name
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_element_size(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_quantized(ggml_type type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		// TODO: temporary until model loading of ggml examples is refactored
		public extern static ggml_type ggml_ftype_to_ggml_type(ggml_ftype ftype);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_transposed(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_contiguous(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_permuted(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_empty(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_scalar(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_vector(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_matrix(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_is_3d(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_n_dims(ggml_tensor* tensor); // returns 1 for scalars

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_are_same_shape(ggml_tensor* t0, ggml_tensor* t1);

		/// <summary>
		/// use this to compute the memory overhead of a tensor
		/// </summary>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_tensor_overhead();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_validate_row_data(ggml_type type, IntPtr data, size_t nbytes);

		// main
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_context* ggml_init(ggml_init_params @params);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_free(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_used_mem(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_set_scratch(ggml_context* ctx, ggml_scratch scratch);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_get_no_alloc(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_no_alloc(ggml_context* ctx, bool no_alloc);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_get_mem_buffer(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_get_mem_size(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_get_max_tensor_size(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_tensor(ggml_context* ctx, ggml_type type, int n_dims, int64_t[] ne);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_tensor_1d(ggml_context* ctx, ggml_type type, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_tensor_2d(ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_tensor_3d(ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_tensor_4d(ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_i32(ggml_context* ctx, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_new_f32(ggml_context* ctx, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_dup_tensor(ggml_context* ctx, ggml_tensor* src);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_view_tensor(ggml_context* ctx, ggml_tensor* src);

		// Context tensor  eration and lookup
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_first_tensor(ggml_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_next_tensor(ggml_context* ctx, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_tensor(ggml_context* ctx, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_zero(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_i32(ggml_tensor* tensor, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_f32(ggml_tensor* tensor, float value);

		// Converts a flat index into coordinates
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_unravel_index(ggml_tensor* tensor, int64_t i, int64_t* i0, int64_t* i1, int64_t* i2, int64_t* i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t ggml_get_i32_1d(ggml_tensor* tensor, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_i32_1d(ggml_tensor* tensor, int i, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t ggml_get_i32_nd(ggml_tensor* tensor, int i0, int i1, int i2, int i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_i32_nd(ggml_tensor* tensor, int i0, int i1, int i2, int i3, int32_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_get_f32_1d(ggml_tensor* tensor, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_f32_1d(ggml_tensor* tensor, int i, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float ggml_get_f32_nd(ggml_tensor* tensor, int i0, int i1, int i2, int i3);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_f32_nd(ggml_tensor* tensor, int i0, int i1, int i2, int i3, float value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_get_data(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float* ggml_get_data_f32(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_unary_op ggml_get_unary_op(ggml_tensor* tensor);

		public static string? ggml_get_name(ggml_tensor* tensor)
		{
			return Marshal.PtrToStringAnsi(ggml_get_name_native(tensor));
			[DllImport(DllName, EntryPoint = "ggml_get_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_get_name_native(ggml_tensor* tensor);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_name(ggml_tensor* tensor, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_format_name(ggml_tensor* tensor, string fmt);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_dup(ggml_context* ctx, ggml_tensor* a);

		// in-place, returns view(a)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_dup_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add_cast(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_type type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add1(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add1_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		// dst = a
		// view(dst, nb1, nb2, nb3, offset) += b
		// return dst
		public extern static ggml_tensor* ggml_acc(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_acc_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sub(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sub_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_mul(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_mul_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_div(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_div_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sqr(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sqr_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sqrt(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sqrt_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_log(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_log_inplace(ggml_context* ctx, ggml_tensor* a);


		/// <summary>
		/// return scalar
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sum(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// sums along rows, with input shape [a,b,c,d] return shape [1,b,c,d]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sum_rows(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// mean along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_mean(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// argmax along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_argmax(ggml_context* ctx, ggml_tensor* a);


		/// <summary>
		/// if a is the same shape as b, and a is not parameter, return a otherwise, return a new tensor: repeat(a) to fit in b
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_repeat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// sums repetitions in a into shape of b
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_repeat_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// concat a and b on dim 2 used in stable-diffusion
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_concat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_abs(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_abs_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sgn(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sgn_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_neg(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_neg_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_step(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_step_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_tanh(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_tanh_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_elu(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_elu_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_relu(ggml_context* ctx, ggml_tensor* a);

		// contains in ggml but not in llama.cpp
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sigmoid(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_leaky_relu(ggml_context* ctx, ggml_tensor* a, float negative_slope, bool inplace);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_relu_inplace(ggml_context* ctx, ggml_tensor* a);

		// contains in ggml but not in llama.cpp
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_sigmoid_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_gelu(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_gelu_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_gelu_quick(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_gelu_quick_inplace(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_silu(ggml_context* ctx, ggml_tensor* a);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_silu_inplace(ggml_context* ctx, ggml_tensor* a);

		// a - x
		// b - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_silu_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// hardswish(x) = x * relu6(x + 3) / 6
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_hardswish(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// hardsigmoid(x) = relu6(x + 3) / 6
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_hardsigmoid(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// normalize along rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="eps"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_norm(ggml_context* ctx, ggml_tensor* a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_norm_inplace(ggml_context* ctx, ggml_tensor* a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_rms_norm(ggml_context* ctx, ggml_tensor* a, float eps);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_rms_norm_inplace(ggml_context* ctx, ggml_tensor* a, float eps);

		/// <summary>
		/// group normalize along ne0*ne1*n_groups
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_groups"></param>
		/// <returns></returns>
		// used in stable-diffusion
		// TODO: eps is hardcoded to 1e-6 for now
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_group_norm(ggml_context* ctx, ggml_tensor* a, int n_groups);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_group_norm_inplace(ggml_context* ctx, ggml_tensor* a, int n_groups);

		// a - x
		// b - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_rms_norm_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, float eps);

		/// <summary>
		/// result is n columns, m rows => [ne03 * x, ne02 * y, m, n]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a">k columns, n rows => [ne03, ne02, n, k]</param>
		/// <param name="b">k columns, m rows  (i.e. we transpose it internally) => [ne03 * x, ne02 * y, m, k]</param>
		/// <returns></returns>
		// 
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_mul_mat(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// change the precision of a matrix multiplication set to GGML_PREC_F32 for higher precision (useful for phi-2)
		/// </summary>
		/// <param name="a"></param>
		/// <param name="prec"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_mul_mat_set_prec(ggml_tensor* a, ggml_prec prec);

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
		public extern static ggml_tensor* ggml_mul_mat_id(ggml_context* ctx, ggml_tensor* @as, ggml_tensor* ids, int id, ggml_tensor* b);


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
		public extern static ggml_tensor* ggml_mul_mat_id(ggml_context* ctx, ggml_tensor* @as, ggml_tensor* ids, ggml_tensor* b);

		/// <summary>
		/// result is m columns, p rows
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a">m columns, n rows,</param>
		/// <param name="b">p columns, n rows,</param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_out_prod(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// operations on tensors without backpropagation
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="s"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_scale(ggml_context* ctx, ggml_tensor* a, float s);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="s"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_scale_inplace(ggml_context* ctx, ggml_tensor* a, float s);

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
		public extern static ggml_tensor* ggml_set(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

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
		public extern static ggml_tensor* ggml_set_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_1d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_1d_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t offset);

		// b -> view(a,offset,nb1,nb2,3), return modified a
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_set_2d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t offset);

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
		public extern static ggml_tensor* ggml_set_2d_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, size_t nb1, size_t offset);

		/// <summary>
		/// a -> b, return view(b)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cpy(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cast(ggml_context* ctx, ggml_tensor* a, ggml_type type);

		/// <summary>
		/// make contiguous
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cont(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// make contiguous, with new shape
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cont_1d(ggml_context* ctx, ggml_tensor* a, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cont_2d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cont_3d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cont_4d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		/// <summary>
		/// return view(a), b specifies the new shape
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		// TODO: when we start computing gradient, make a copy instead of view
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_reshape(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// return view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <returns></returns>
		// TODO: when we start computing gradient, make a copy instead of view
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_reshape_1d(ggml_context* ctx, ggml_tensor* a, int64_t ne0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_reshape_2d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1);

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
		public extern static ggml_tensor* ggml_reshape_3d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_reshape_4d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);

		/// <summary>
		/// offset in bytes
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="ne0"></param>
		/// <param name="offset"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_view_1d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, size_t offset);

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
		public extern static ggml_tensor* ggml_view_2d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, size_t nb1, size_t offset);

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
		public extern static ggml_tensor* ggml_view_3d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, size_t nb1, size_t nb2, size_t offset);

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
		public extern static ggml_tensor* ggml_view_4d(ggml_context* ctx, ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, size_t nb1, size_t nb2, size_t nb3, size_t offset);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_permute(ggml_context* ctx, ggml_tensor* a, int axis0, int axis1, int axis2, int axis3);

		/// <summary>
		/// alias for ggml_permute(ctx, a, 1, 0, 2, 3)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_transpose(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// supports 3D: a->ne[2] == b->ne[1]
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_rows(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_rows_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_diag(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// set elements above the diagonal to -INF
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_diag_mask_inf(ggml_context* ctx, ggml_tensor* a, int n_past);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_diag_mask_inf_inplace(ggml_context* ctx, ggml_tensor* a, int n_past);

		/// <summary>
		/// set elements above the diagonal to 0
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_diag_mask_zero(ggml_context* ctx, ggml_tensor* a, int n_past);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="n_past"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_diag_mask_zero_inplace(ggml_context* ctx, ggml_tensor* a, int n_past);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_soft_max(ggml_context* ctx, ggml_tensor* a);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_soft_max_inplace(ggml_context* ctx, ggml_tensor* a);

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
		public extern static ggml_tensor* ggml_soft_max_ext(ggml_context* ctx, ggml_tensor* a, ggml_tensor* mask, float scale, float max_bias);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_soft_max_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		/// <summary>
		/// in-place, returns view(a)
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_soft_max_back_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		// rotary position embedding
		// if mode & 1 == 1, skip n_past elements (DEPRECATED)
		// if mode & 2 == 1, GPT-NeoX style
		// if mode & 4 == 1, ChatGLM style
		//
		// b is an int32 vector with size a->ne[2], it contains the positions
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_rope(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, int mode, int n_ctx);

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
		public extern static ggml_tensor* ggml_rope_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, int mode, int n_ctx);

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
		public extern static ggml_tensor* ggml_rope_custom(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

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
		public extern static ggml_tensor* ggml_rope_custom_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow);

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
		public extern static ggml_tensor* ggml_rope_xpos_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, float @base, bool down);

		// rotary position embedding backward, i.e compute dx from dy
		// a - dy
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_rope_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int n_dims, int mode, int n_ctx, int n_orig_ctx, float freq_base, float freq_scale, float ext_factor, float attn_factor, float beta_fast, float beta_slow, float xpos_base, bool xpos_down);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_clamp(ggml_context* ctx, ggml_tensor* a, float min, float max);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_im2col(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D, ggml_type dst_type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_depthwise_2d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1);

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
		public extern static ggml_tensor* ggml_conv_1d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s0, int p0, int d0);

		// conv_1d with padding = half
		// alias for ggml_conv_1d(a, b, s, a->ne[0]/2, d)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_1d_ph(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s, int d);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]

		public extern static ggml_tensor* ggml_conv_transpose_1d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s0, int p0, int d0);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_2d(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int s0, int s1, int p0, int p1, int d0, int d1);


		// kernel size is a->ne[0] x a->ne[1]
		// stride is equal to kernel size
		// padding is zero
		// example:
		// a:     16   16    3  768
		// b:   1024 1024    3    1
		// res:   64   64  768    1
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_2d_sk_p0(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		// kernel size is a->ne[0] x a->ne[1]
		// stride is 1
		// padding is half
		// example:
		// a:      3    3    256  256
		// b:     64   64    256    1
		// res:   64   64    256    1
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_2d_s1_ph(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_conv_transpose_2d_p0(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, int stride);

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
		public extern static ggml_tensor* ggml_pool_1d(ggml_context* ctx, ggml_tensor* a, ggml_op_pool op, int k0, int s0, int p0);

		// the result will have 2*p0 padding for the first dimension
		// and 2*p1 padding for the second dimension
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_pool_2d(ggml_context* ctx, ggml_tensor* a, ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1);

		// nearest interpolate
		// used in stable-diffusion
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_upscale(ggml_context* ctx, ggml_tensor* a, int scale_factor);

		// pad each dimension with zeros: [x, ..., x] -> [x, ..., x, 0, ..., 0]
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_pad(ggml_context* ctx, ggml_tensor* a, int p0, int p1, int p2, int p3);

		// Ref: https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/util.py#L151
		// timesteps: [N,]
		// return: [N, dim]
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_timestep_embedding(ggml_context* ctx, ggml_tensor* timesteps, int dim, int max_period);

		// sort rows

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_argsort(ggml_context* ctx, ggml_tensor* a, ggml_sort_order order);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_arange(ggml_context* ctx, float start, float stop, float step);

		// top k elements per row
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_top_k(ggml_context* ctx, ggml_tensor* a, int k);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_flash_attn(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, bool masked);

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
		public extern static ggml_tensor* ggml_flash_attn_ext(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* mask, float scale, float max_bias);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_flash_attn_ext_set_prec(ggml_tensor* a, ggml_prec prec);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_flash_attn_back(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v, ggml_tensor* d, bool masked);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_flash_ff(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b0, ggml_tensor* b1, ggml_tensor* c0, ggml_tensor* c1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_ssm_conv(ggml_context* ctx, ggml_tensor* s, ggml_tensor* x, ggml_tensor* c, ggml_tensor* sq);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_ssm_scan(ggml_context* ctx, ggml_tensor* s, ggml_tensor* x, ggml_tensor* dt, ggml_tensor* A, ggml_tensor* B, ggml_tensor* C, ggml_tensor* sq);

		// partition into non-overlapping windows with padding if needed
		// example:
		// a:   768   64   64    1
		// w:    14
		// res: 768   14   14    25
		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_win_part(ggml_context* ctx, ggml_tensor* a, int w);

		/// <summary>
		/// reverse of ggml_win_part used in sam
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="a"></param>
		/// <param name="w0"></param>
		/// <param name="h0"></param>
		/// <param name="w"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_win_unpart(ggml_context* ctx, ggml_tensor* a, int w0, int h0, int w);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_unary(ggml_context* ctx, ggml_tensor* a, ggml_unary_op op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_unary_inplace(ggml_context* ctx, ggml_tensor* a, ggml_unary_op op);

		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_get_rel_pos(ggml_context* ctx, ggml_tensor* a, int qh, int kh);

		// used in sam
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add_rel_pos(ggml_context* ctx, ggml_tensor* a, ggml_tensor* pw, ggml_tensor* ph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_add_rel_pos_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* pw, ggml_tensor* ph);

		public delegate void ggml_custom1_op_t(ggml_tensor* dst, ggml_tensor* a, int ith, int nth, void* userdata);
		public delegate void ggml_custom2_op_t(ggml_tensor* dst, ggml_tensor* a, ggml_tensor* b, int ith, int nth, void* userdata);
		public delegate void ggml_custom3_op_t(ggml_tensor* dst, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, int ith, int nth, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom1(ggml_context* ctx, ggml_tensor* a, ggml_custom1_op_t fun, int n_tasks, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom1_inplace(ggml_context* ctx, ggml_tensor* a, ggml_custom1_op_t fun, int n_tasks, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom2(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_custom2_op_t fun, int n_tasks, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom2_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_custom2_op_t fun, int n_tasks, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom3(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, ggml_custom3_op_t fun, int n_tasks, void* userdata);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_map_custom3_inplace(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c, ggml_custom3_op_t fun, int n_tasks, void* userdata);

		// loss function

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cross_entropy_loss(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_cross_entropy_loss_back(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* c);

		//
		// automatic differentiation
		//

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_param(ggml_context* ctx, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_forward_expand(ggml_cgraph* cgraph, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_backward_expand(ggml_context* ctx, ggml_cgraph* gf, ggml_cgraph* gb, bool keep);

		// graph allocation in a context
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph* ggml_new_graph(ggml_context* ctx); // size = GGML_DEFAULT_GRAPH_SIZE, grads = false

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph* ggml_new_graph_custom(ggml_context* ctx, size_t size, bool grads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph* ggml_graph_dup(ggml_context* ctx, ggml_cgraph* cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph ggml_graph_view(ggml_cgraph* cgraph, int i0, int i1);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_cpy(ggml_cgraph* src, ggml_cgraph* dst);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_reset(ggml_cgraph* cgraph);  // zero grads

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_clear(ggml_cgraph* cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_graph_overhead();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_graph_overhead_custom(size_t size, bool grads);

		// ggml_graph_plan() has to be called before ggml_graph_compute()
		// when plan.work_size > 0, caller must allocate memory for plan.work_data
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cplan ggml_graph_plan(ggml_cgraph* cgraph, int n_threads /*= GGML_DEFAULT_N_THREADS*/);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_graph_compute(ggml_cgraph* cgraph, ggml_cplan* cplan);
		// same as ggml_graph_compute() but the work data is allocated as a part of the context
		// note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_graph_compute_with_ctx(ggml_context* ctx, ggml_cgraph* cgraph, int n_threads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tensor* ggml_graph_get_tensor(ggml_cgraph* cgraph, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_export(ggml_cgraph* cgraph, string fname);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_cgraph* ggml_graph_import(string fname, ggml_context** ctx_data, ggml_context** ctx_eval);

		// print info and performance information for the graph
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_print(ggml_cgraph* cgraph);

		// dump the graph into a file using the dot format
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_graph_dump_dot(ggml_cgraph* gb, ggml_cgraph* gf, string filename);

		// build gradient checkpointing backward graph gb for gf using provided checkpoints
		// gb_tmp will contain original backward graph with rewritten backward process nodes,
		// but without the second forward pass nodes.
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_build_backward_gradient_checkpointing(ggml_context* ctx, ggml_cgraph* gf, ggml_cgraph* gb, ggml_cgraph* gb_tmp, ggml_tensor** checkpoints, int n_checkpoints);

		public delegate void ggml_opt_callback(void* data, int accum_step, float* sched, bool* cancel);
		public delegate void ggml_log_callback(ggml_log_level level, string text, void* user_data);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_params ggml_opt_default_params(ggml_opt_type type);

		// optimize the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_result ggml_opt(ggml_context* ctx, ggml_opt_params @params, ggml_tensor* f);

		// initialize optimizer context
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_opt_init(ggml_context* ctx, ggml_opt_context* opt, ggml_opt_params @params, int64_t nx);

		// continue optimizing the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_result ggml_opt_resume(ggml_context* ctx, ggml_opt_context* opt, ggml_tensor* f);

		// continue optimizing the function defined by the tensor f
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_opt_result ggml_opt_resume_g(ggml_context* ctx, ggml_opt_context* opt, ggml_tensor* f, ggml_cgraph* gf, ggml_cgraph* gb, ggml_opt_callback callback, void* callback_data);

		//
		// tensor flags
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_input(ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_set_output(ggml_tensor* tensor);

		//
		// quantization
		//

		// - ggml_quantize_init can be called multiple times with the same type
		//   it will only initialize the quantization tables for the first call or after ggml_quantize_free
		//   automatically called by ggml_quantize_chunk for convenience
		//
		// - ggml_quantize_free will free any memory allocated by ggml_quantize_init
		//   call this at the end of the program to avoid memory leaks
		//
		// note: these are thread-safe
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_quantize_init(ggml_type type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_quantize_free();

		// some quantization type cannot be used without an importance matrix
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_quantize_requires_imatrix(ggml_type type);

		// calls ggml_quantize_init internally (i.e. can allocate memory)
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_quantize_chunk(ggml_type type, float* src, void* dst, int64_t start, int64_t nrows, int64_t n_per_row, float* imatrix);

		public static size_t gguf_type_size(gguf_type type)
		{
			switch (type)
			{
				case gguf_type.GGUF_TYPE_UINT8:
					return sizeof(uint8_t);
				case gguf_type.GGUF_TYPE_INT8:
					return sizeof(int8_t);
				case gguf_type.GGUF_TYPE_UINT16:
					return sizeof(uint16_t);
				case gguf_type.GGUF_TYPE_INT16:
					return sizeof(int16_t);
				case gguf_type.GGUF_TYPE_UINT32:
					return sizeof(uint32_t);
				case gguf_type.GGUF_TYPE_INT32:
					return sizeof(int32_t);
				case gguf_type.GGUF_TYPE_FLOAT32:
					return sizeof(float);
				case gguf_type.GGUF_TYPE_BOOL:
					return sizeof(bool);
				case gguf_type.GGUF_TYPE_STRING:
					return sizeof(gguf_str);
				case gguf_type.GGUF_TYPE_UINT64:
					return sizeof(uint64_t);
				case gguf_type.GGUF_TYPE_INT64:
					return sizeof(int64_t);
				case gguf_type.GGUF_TYPE_FLOAT64:
					return sizeof(double);
				case gguf_type.GGUF_TYPE_ARRAY:
					return 0; // undefined
				default:
					return 0;
			}
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static gguf_context* gguf_init_empty();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static gguf_context* gguf_init_from_file(string fname, gguf_init_params @params);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_free(gguf_context* ctx);

		public static string? gguf_type_name(gguf_type type)
		{
			return Marshal.PtrToStringAnsi(gguf_type_name_native(type));
			[DllImport(DllName, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gguf_type_name")]
			extern static IntPtr gguf_type_name_native(gguf_type type);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_version(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_alignment(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_data_offset(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_data(gguf_context* ctx);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_n_kv(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool gguf_find_key(gguf_context* ctx, string key);


		public static string gguf_get_key(gguf_context* ctx, int key_id)
		{
			return Marshal.PtrToStringAnsi(gguf_get_key_native(ctx, key_id));
			[DllImport(DllName, EntryPoint = "gguf_get_key", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr gguf_get_key_native(gguf_context* ctx, int key_id);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static gguf_type gguf_get_kv_type(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static gguf_type gguf_get_arr_type(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_arr_data(gguf_context* ctx, int key_id);

		public static string gguf_get_arr_str(gguf_context* ctx, int key_id, int i)
		{
			return Marshal.PtrToStringAnsi(gguf_get_arr_str_native(ctx, key_id, i));
			[DllImport(DllName, EntryPoint = "gguf_get_arr_str", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr gguf_get_arr_str_native(gguf_context* ctx, int key_id, int i);
		}


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_arr_n(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint8_t gguf_get_val_u8(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int8_t gguf_get_val_i8(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint16_t gguf_get_val_u16(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int16_t gguf_get_val_i16(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint32_t gguf_get_val_u32(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int32_t gguf_get_val_i32(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static float gguf_get_val_f32(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static uint64_t gguf_get_val_u64(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int64_t gguf_get_val_i64(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static double gguf_get_val_f64(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool gguf_get_val_bool(gguf_context* ctx, int key_id);


		public static string? gguf_get_val_str(gguf_context* ctx, int key_id)
		{
			return Marshal.PtrToStringAnsi(gguf_get_val_str_native(ctx, key_id));
			[DllImport(DllName, EntryPoint = "gguf_get_val_str", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr gguf_get_val_str_native(gguf_context* ctx, int key_id);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* gguf_get_val_data(gguf_context* ctx, int key_id);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_n_tensors(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_find_tensor(gguf_context* ctx, string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_tensor_offset(gguf_context* ctx, int i);

		public static string? gguf_get_tensor_name(gguf_context* ctx, int i)
		{
			return Marshal.PtrToStringAnsi(gguf_get_tensor_name_native(ctx, i));
			[DllImport(DllName, EntryPoint = "gguf_get_tensor_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr gguf_get_tensor_name_native(gguf_context* ctx, int i);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_type gguf_get_tensor_type(gguf_context* ctx, int i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_remove_key(gguf_context* ctx, string key);

		// returns the index
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int gguf_get_or_add_key(gguf_context* ctx, string key);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u8(gguf_context* ctx, string key, uint8_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i8(gguf_context* ctx, string key, int8_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u16(gguf_context* ctx, string key, uint16_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i16(gguf_context* ctx, string key, int16_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u32(gguf_context* ctx, string key, uint32_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i32(gguf_context* ctx, string key, int32_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_f32(gguf_context* ctx, string key, float val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_u64(gguf_context* ctx, string key, uint64_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_i64(gguf_context* ctx, string key, int64_t val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_f64(gguf_context* ctx, string key, double val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_bool(gguf_context* ctx, string key, bool val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_val_str(gguf_context* ctx, string key, string val);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_arr_data(gguf_context* ctx, string key, gguf_type type, IntPtr data, int n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		//public extern static void gguf_set_arr_str(gguf_context* ctx, string key, IntPtr data, int n);
		public extern static void gguf_set_arr_str(gguf_context* ctx, string key, string[] data, int n);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		//public extern static void gguf_set_arr_str(gguf_context* ctx, string key, IntPtr data, int n);
		public extern static void gguf_set_arr_str(gguf_context* ctx, string key, IntPtr[] data, int n);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public static extern void gguf_set_kv(gguf_context* ctx, gguf_context* src);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_add_tensor(gguf_context* ctx, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_tensor_type(gguf_context* ctx, string name, ggml_type type);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_set_tensor_data(gguf_context* ctx, string name, IntPtr data, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_write_to_file(gguf_context* ctx, string fname, bool only_meta);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t gguf_get_meta_size(gguf_context* ctx);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void gguf_get_meta_data(gguf_context* ctx, IntPtr data);

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

		public delegate ggml_backend_t ggml_backend_init_fn(string @params, IntPtr user_data);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_buffer_init(ggml_backend_buffer_type_t buft, ggml_backend_buffer_i iface, ggml_backend_buffer_context_t context, size_t size);

		/// <summary>
		/// do not use directly, use ggml_backend_tensor_copy instead
		/// </summary>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_copy_tensor(ggml_tensor* src, ggml_tensor* dst);

		/// <summary>
		/// buffer that contains a collection of buffers
		/// </summary>
		/// <param name="buffers"></param>
		/// <param name="n_buffers"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t* buffers, size_t n_buffers);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, ggml_backend_buffer_usage usage);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_register(string name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, IntPtr user_data);

		#endregion


		#region ggml-backend.h

		// buffer type
		public static string? ggml_backend_buft_name(ggml_backend_buffer_type_t buft)
		{
			return Marshal.PtrToStringAnsi(ggml_backend_buft_name_native(buft));
			[DllImport(DllName, EntryPoint = "ggml_backend_buft_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_backend_buft_name_native(ggml_backend_buffer_type_t buft);
		}


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buft_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft);


		/// <summary>
		/// Copy a graph to a different backend
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_graph_copy_free(ggml_backend_graph_copy copy);

		public static string? ggml_backend_buffer_name(ggml_backend_buffer_t buffer)
		{
			return Marshal.PtrToStringAnsi(ggml_backend_buffer_name_native(buffer));
			[DllImport(DllName, EntryPoint = "ggml_backend_buffer_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_backend_buffer_name_native(ggml_backend_buffer_t buffer);

		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_free(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void* ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, ggml_backend_buffer_usage usage);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer);

		//
		// Backend
		//

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_guid_t ggml_backend_guid(ggml_backend_t backend);
		public static string? ggml_backend_name(ggml_backend_t backend)
		{
			return Marshal.PtrToStringAnsi(ggml_backend_name_native(backend));
			[DllImport(DllName, EntryPoint = "ggml_backend_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_backend_name_native(ggml_backend_t backend);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_free(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_get_alignment(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_get_max_size(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_set_async(ggml_backend_t backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_get_async(ggml_backend_t backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_set(ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_get(ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_synchronize(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, ggml_cgraph* cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph* cgraph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_supports_op(ggml_backend_t backend, ggml_tensor* op);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_offload_op(ggml_backend_t backend, ggml_tensor* op);

		/// <summary>
		/// tensor copy between different backends
		/// </summary>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_copy(ggml_tensor* src, ggml_tensor* dst);

		// asynchronous copy
		// the copy is performed after all the currently queued operations in backend_src
		// backend_dst will wait for the copy to complete before performing other operations
		// automatic fallback to sync copy if async is not supported
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ggml_tensor* src, ggml_tensor* dst);

		// events
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_event_t ggml_backend_event_new(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_free(ggml_backend_event_t @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_record(ggml_backend_event_t @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_synchronize(ggml_backend_event_t @event);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t @event); // wait async on event

		//
		// CPU backend
		//
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_t ggml_backend_cpu_init();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_is_cpu(ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void* abort_callback_data);

		/// <summary>
		/// Create a backend buffer from an existing pointer
		/// </summary>
		/// <param name="ptr"></param>
		/// <param name="size"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(IntPtr ptr, size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_reg_get_count();

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_reg_find_by_name(string name);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_t ggml_backend_reg_init_backend_from_str(string backend_str); // str is name[:params]

		public static string? ggml_backend_reg_get_name(size_t i)
		{
			return Marshal.PtrToStringAnsi(ggml_backend_reg_get_name_native(i));
			[DllImport(DllName, EntryPoint = "ggml_backend_reg_get_name", CallingConvention = CallingConvention.Cdecl)]
			extern static IntPtr ggml_backend_reg_get_name_native(size_t i);
		}

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_t ggml_backend_reg_init_backend(size_t i, string @params); // params is backend-specific

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer_t ggml_backend_reg_alloc_buffer(size_t i, size_t size);

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
		public extern static ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t* backends, ggml_backend_buffer_type_t* bufts, int n_backends, size_t graph_size, bool parallel);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_free(ggml_backend_sched_t sched);

		/// <summary>
		/// Initialize backend buffers from a measure graph
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="measure_graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, ggml_cgraph* measure_graph);

		/// <summary>
		/// Get the number of splits of the last graph
		/// </summary>
		/// <param name="sched"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, ggml_tensor* node, ggml_backend_t backend);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, ggml_tensor* node);

		/// <summary>
		/// Allocate and compute graph on the backend scheduler
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="graph"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_synchronize(ggml_backend_sched_t sched);

		/// <summary>
		/// Reset all assignments and allocators - must be called before changing the node backends
		/// </summary>
		/// <param name="sched"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_reset(ggml_backend_sched_t sched);

		/// <summary>
		/// Set a callback to be called for each resulting node during graph compute
		/// </summary>
		/// <param name="sched"></param>
		/// <param name="callback"></param>
		/// <param name="user_data"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void* user_data);

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
		public extern static bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, ggml_cgraph* graph, ggml_backend_eval_callback callback, void* user_data);

		/// <summary>
		/// Tensor initialization
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="tensor"></param>
		/// <param name="addr"></param>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, ggml_tensor* tensor, void* addr);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_backend_view_init(ggml_backend_buffer_t buffer, ggml_tensor* tensor);



		#endregion


		#region ggml_alloc.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_tallocr_alloc(ggml_tallocr* talloc, ggml_tensor* tensor);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t* bufts, int n_bufs);
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static void ggml_gallocr_free(ggml_gallocr_t galloc);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_reserve(ggml_gallocr_t galloc, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_reserve_n(ggml_gallocr_t galloc, ggml_cgraph* graph, int* node_buffer_ids, int* leaf_buffer_ids);


		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, ggml_cgraph* graph);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

		/// <summary>
		/// Create a buffer and allocate all the tensors in a ggml_context
		/// </summary>
		/// <param name="ctx"></param>
		/// <param name="buft"></param>
		/// <returns></returns>
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer* ggml_backend_alloc_ctx_tensors_from_buft(ggml_context* ctx, ggml_backend_buffer_type_t buft);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_buffer* ggml_backend_alloc_ctx_tensors(ggml_context* ctx, ggml_backend_t backend);


		#endregion

		#region ggml-impl.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_hash_set ggml_hash_set_new(size_t size);

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static bool ggml_hash_contains(ggml_hash_set hash_set, ggml_tensor* key);

		// returns GGML_HASHTABLE_FULL if table is full, otherwise the current index of the key or where it should be inserted
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_find(ggml_hash_set hash_set, ggml_tensor* key);

		// returns GGML_HASHTABLE_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_insert(ggml_hash_set hash_set, ggml_tensor* key);

		// return index, asserts if table is full
		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static size_t ggml_hash_find_or_insert(ggml_hash_set hash_set, ggml_tensor* key);


		#endregion

		#region ggml-cuda.h

		[DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
		public extern static ggml_backend_t ggml_backend_cuda_init(int device);

		#endregion

	}
}

