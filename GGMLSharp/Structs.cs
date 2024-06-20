using System;
using System.Runtime.InteropServices;

namespace GGMLSharp
{
	public unsafe class Structs
	{
		/// <summary>
		/// "ggml"
		/// </summary>
		const int GGML_FILE_MAGIC = 0x67676d6c;
		const int GGML_FILE_VERSION = 1;

		/// <summary>
		/// bump this on quantization format changes
		/// </summary>
		const int GGML_QNT_VERSION = 2;

		/// <summary>
		/// do not change this
		/// </summary>
		const int GGML_QNT_VERSION_FACTOR = 1000;

		const int GGML_MAX_DIMS = 4;
		const int GGML_MAX_PARAMS = 2048;
		const int GGML_MAX_CONTEXTS = 64;
		const int GGML_MAX_SRC = 10;
		const int GGML_MAX_NAME = 128; // 64?
		const int GGML_MAX_OP_PARAMS = 64;
		const int GGML_DEFAULT_N_THREADS = 4;
		public const int GGML_DEFAULT_GRAPH_SIZE = 2048;

		/// <summary>
		///  x64 only
		/// </summary>
		const int GGML_MEM_ALIGN = 16;

		const int GGML_EXIT_SUCCESS = 0;
		const int GGML_EXIT_ABORTED = 1;

		const string GGUF_MAGIC = "GGUF";

		const int GGUF_VERSION = 3;

		const int GGUF_DEFAULT_ALIGNMENT = 32;
		const int GGML_N_TASKS_MAX = -1;
		const int GGML_KQ_MASK_PAD = 32;
		const int MAX_FREE_BLOCKS = 256;


		#region ggml.h

		public enum ggml_status
		{
			GGML_STATUS_ALLOC_FAILED = -2,
			GGML_STATUS_FAILED = -1,
			GGML_STATUS_SUCCESS = 0,
			GGML_STATUS_ABORTED = 1,
		};
		public enum ggml_type
		{
			GGML_TYPE_F32 = 0,
			GGML_TYPE_F16 = 1,
			GGML_TYPE_Q4_0 = 2,
			GGML_TYPE_Q4_1 = 3,
			// GGML_TYPE_Q4_2 = 4, support has been removed
			// GGML_TYPE_Q4_3 = 5, support has been removed
			GGML_TYPE_Q5_0 = 6,
			GGML_TYPE_Q5_1 = 7,
			GGML_TYPE_Q8_0 = 8,
			GGML_TYPE_Q8_1 = 9,
			GGML_TYPE_Q2_K = 10,
			GGML_TYPE_Q3_K = 11,
			GGML_TYPE_Q4_K = 12,
			GGML_TYPE_Q5_K = 13,
			GGML_TYPE_Q6_K = 14,
			GGML_TYPE_Q8_K = 15,
			GGML_TYPE_IQ2_XXS = 16,
			GGML_TYPE_IQ2_XS = 17,
			GGML_TYPE_IQ3_XXS = 18,
			GGML_TYPE_IQ1_S = 19,
			GGML_TYPE_IQ4_NL = 20,
			GGML_TYPE_IQ3_S = 21,
			GGML_TYPE_IQ2_S = 22,
			GGML_TYPE_IQ4_XS = 23,
			GGML_TYPE_I8 = 24,
			GGML_TYPE_I16 = 25,
			GGML_TYPE_I32 = 26,
			GGML_TYPE_I64 = 27,
			GGML_TYPE_F64 = 28,
			GGML_TYPE_IQ1_M = 29,
			GGML_TYPE_BF16 = 30,
			GGML_TYPE_COUNT,
		};

		/// <summary>
		/// precision
		/// </summary>
		public enum ggml_prec
		{
			GGML_PREC_DEFAULT,
			GGML_PREC_F32,
		};

		public enum ggml_backend_type
		{
			GGML_BACKEND_TYPE_CPU = 0,
			GGML_BACKEND_TYPE_GPU = 10,
			GGML_BACKEND_TYPE_GPU_SPLIT = 20,
		};

		/// <summary>
		/// model file types
		/// </summary>
		public enum ggml_ftype
		{
			GGML_FTYPE_UNKNOWN = -1,
			GGML_FTYPE_ALL_F32 = 0,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_F16 = 1,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_0 = 2,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_1 = 3,

			/// <summary>
			/// tok_embeddings.weight and output.weight are F16
			/// </summary>
			GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q8_0 = 7,

			/// <summary>
			/// except 1d tensors
			/// </summary>
			GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
			GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
			GGML_FTYPE_MOSTLY_Q2_K = 10, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q3_K = 11, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q4_K = 12, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q5_K = 13, // except 1d tensors
			GGML_FTYPE_MOSTLY_Q6_K = 14, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_XS = 16, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ1_S = 18, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ4_NL = 19, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ3_S = 20, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ2_S = 21, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ4_XS = 22, // except 1d tensors
			GGML_FTYPE_MOSTLY_IQ1_M = 23, // except 1d tensors
			GGML_FTYPE_MOSTLY_BF16 = 24, // except 1d tensors
		};

		/// <summary>
		/// available tensor operations:
		/// </summary>
		public enum ggml_op
		{
			GGML_OP_NONE = 0,

			GGML_OP_DUP,
			GGML_OP_ADD,
			GGML_OP_ADD1,
			GGML_OP_ACC,
			GGML_OP_SUB,
			GGML_OP_MUL,
			GGML_OP_DIV,
			GGML_OP_SQR,
			GGML_OP_SQRT,
			GGML_OP_LOG,
			GGML_OP_SUM,
			GGML_OP_SUM_ROWS,
			GGML_OP_MEAN,
			GGML_OP_ARGMAX,
			GGML_OP_REPEAT,
			GGML_OP_REPEAT_BACK,
			GGML_OP_CONCAT,
			GGML_OP_SILU_BACK,
			/// <summary>
			/// normalize
			/// </summary>
			GGML_OP_NORM,
			GGML_OP_RMS_NORM,
			GGML_OP_RMS_NORM_BACK,
			GGML_OP_GROUP_NORM,

			GGML_OP_MUL_MAT,
			GGML_OP_MUL_MAT_ID,
			GGML_OP_OUT_PROD,

			GGML_OP_SCALE,
			GGML_OP_SET,
			GGML_OP_CPY,
			GGML_OP_CONT,
			GGML_OP_RESHAPE,
			GGML_OP_VIEW,
			GGML_OP_PERMUTE,
			GGML_OP_TRANSPOSE,
			GGML_OP_GET_ROWS,
			GGML_OP_GET_ROWS_BACK,
			GGML_OP_DIAG,
			GGML_OP_DIAG_MASK_INF,
			GGML_OP_DIAG_MASK_ZERO,
			GGML_OP_SOFT_MAX,
			GGML_OP_SOFT_MAX_BACK,
			GGML_OP_ROPE,
			GGML_OP_ROPE_BACK,
			GGML_OP_CLAMP,
			GGML_OP_CONV_TRANSPOSE_1D,
			GGML_OP_IM2COL,
			GGML_OP_CONV_TRANSPOSE_2D,
			GGML_OP_POOL_1D,
			GGML_OP_POOL_2D,
			/// <summary>
			/// nearest interpolate
			/// </summary>
			GGML_OP_UPSCALE,
			GGML_OP_PAD,
			GGML_OP_ARANGE,
			GGML_OP_TIMESTEP_EMBEDDING,
			GGML_OP_ARGSORT,
			GGML_OP_LEAKY_RELU,

			GGML_OP_FLASH_ATTN,
			GGML_OP_FLASH_ATTN_EXT,
			GGML_OP_FLASH_FF,
			GGML_OP_FLASH_ATTN_BACK,
			GGML_OP_SSM_CONV,
			GGML_OP_SSM_SCAN,
			GGML_OP_WIN_PART,
			GGML_OP_WIN_UNPART,
			GGML_OP_GET_REL_POS,
			GGML_OP_ADD_REL_POS,

			GGML_OP_UNARY,

			GGML_OP_MAP_UNARY,
			GGML_OP_MAP_BINARY,

			GGML_OP_MAP_CUSTOM1_F32,
			GGML_OP_MAP_CUSTOM2_F32,
			GGML_OP_MAP_CUSTOM3_F32,

			GGML_OP_MAP_CUSTOM1,
			GGML_OP_MAP_CUSTOM2,
			GGML_OP_MAP_CUSTOM3,

			GGML_OP_CROSS_ENTROPY_LOSS,
			GGML_OP_CROSS_ENTROPY_LOSS_BACK,

			GGML_OP_COUNT,
		};

		public enum ggml_unary_op
		{
			GGML_UNARY_OP_ABS,
			GGML_UNARY_OP_SGN,
			GGML_UNARY_OP_NEG,
			GGML_UNARY_OP_STEP,
			GGML_UNARY_OP_TANH,
			GGML_UNARY_OP_ELU,
			GGML_UNARY_OP_RELU,
			//GGML_UNARY_OP_SIGMOID,  //exist in ggml but in llama.cpp doesn't
			GGML_UNARY_OP_GELU,
			GGML_UNARY_OP_GELU_QUICK,
			GGML_UNARY_OP_SILU,
			GGML_UNARY_OP_HARDSWISH,
			GGML_UNARY_OP_HARDSIGMOID,

			GGML_UNARY_OP_COUNT,
		};

		public enum ggml_object_type
		{
			GGML_OBJECT_TYPE_TENSOR,
			GGML_OBJECT_TYPE_GRAPH,
			GGML_OBJECT_TYPE_WORK_BUFFER
		};

		public enum ggml_log_level
		{
			GGML_LOG_LEVEL_ERROR = 2,
			GGML_LOG_LEVEL_WARN = 3,
			GGML_LOG_LEVEL_INFO = 4,
			GGML_LOG_LEVEL_DEBUG = 5
		};

		public enum ggml_tensor_flag
		{
			GGML_TENSOR_FLAG_INPUT = 1,
			GGML_TENSOR_FLAG_OUTPUT = 2,
			GGML_TENSOR_FLAG_PARAM = 4,
		};

		public enum ggml_task_type
		{
			GGML_TASK_TYPE_INIT = 0,
			GGML_TASK_TYPE_COMPUTE,
			GGML_TASK_TYPE_FINALIZE,
		};

		public enum ggml_cgraph_eval_order
		{
			GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
			GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
			GGML_CGRAPH_EVAL_ORDER_COUNT
		};

		public enum ggml_numa_strategy
		{
			GGML_NUMA_STRATEGY_DISABLED = 0,
			GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
			GGML_NUMA_STRATEGY_ISOLATE = 2,
			GGML_NUMA_STRATEGY_NUMACTL = 3,
			GGML_NUMA_STRATEGY_MIRROR = 4,
			GGML_NUMA_STRATEGY_COUNT
		};

		public enum ggml_sort_order
		{
			GGML_SORT_ORDER_ASC,
			GGML_SORT_ORDER_DESC,
		};

		public enum ggml_op_pool
		{
			GGML_OP_POOL_MAX,
			GGML_OP_POOL_AVG,
			GGML_OP_POOL_COUNT,
		};

		public enum ggml_opt_type
		{
			GGML_OPT_TYPE_ADAM,
			GGML_OPT_TYPE_LBFGS,
		};

		/// <summary>
		/// linesearch methods
		/// </summary>
		public enum ggml_linesearch
		{
			GGML_LINESEARCH_DEFAULT = 1,

			GGML_LINESEARCH_BACKTRACKING_ARMIJO = 0,
			GGML_LINESEARCH_BACKTRACKING_WOLFE = 1,
			GGML_LINESEARCH_BACKTRACKING_STRONG_WOLFE = 2,
		};

		/// <summary>
		/// optimization return values
		/// </summary>
		public enum ggml_opt_result
		{
			GGML_OPT_RESULT_OK = 0,
			GGML_OPT_RESULT_DID_NOT_CONVERGE,
			GGML_OPT_RESULT_NO_CONTEXT,
			GGML_OPT_RESULT_INVALID_WOLFE,
			GGML_OPT_RESULT_FAIL,
			GGML_OPT_RESULT_CANCEL,

			GGML_LINESEARCH_FAIL = -128,
			GGML_LINESEARCH_MINIMUM_STEP,
			GGML_LINESEARCH_MAXIMUM_STEP,
			GGML_LINESEARCH_MAXIMUM_ITERATIONS,
			GGML_LINESEARCH_INVALID_PARAMETERS,
		};

		//
		// gguf
		//
		public enum gguf_type
		{
			GGUF_TYPE_UINT8 = 0,
			GGUF_TYPE_INT8 = 1,
			GGUF_TYPE_UINT16 = 2,
			GGUF_TYPE_INT16 = 3,
			GGUF_TYPE_UINT32 = 4,
			GGUF_TYPE_INT32 = 5,
			GGUF_TYPE_FLOAT32 = 6,
			GGUF_TYPE_BOOL = 7,
			GGUF_TYPE_STRING = 8,
			GGUF_TYPE_ARRAY = 9,
			GGUF_TYPE_UINT64 = 10,
			GGUF_TYPE_INT64 = 11,
			GGUF_TYPE_FLOAT64 = 12,
			/// <summary>
			/// marks the end of the enum
			/// </summary>
			GGUF_TYPE_COUNT,
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_object
		{
			public size_t offs;
			public size_t size;

			public ggml_object* next;

			public ggml_object_type type;

			public fixed byte padding[4];
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_tensor
		{
			public ggml_type type;
			public ggml_backend_type backend;

			public ggml_backend_buffer* buffer;

			/// <summary>
			/// number of elements
			/// </summary>
			public fixed int64_t ne[GGML_MAX_DIMS];
			public fixed size_t nb[GGML_MAX_DIMS]; // stride in bytes:
												   // nb[0] = ggml_type_size(type)
												   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
												   // nb[i] = nb[i-1] * ne[i-1]

			/// <summary>
			/// compute data
			/// </summary>
			public ggml_op op;

			/// <summary>
			/// op params - allocated as int32_t for alignment
			/// </summary>
			public fixed int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

			public int32_t flags;

			public ggml_tensor* grad;
			public fixed long src[GGML_MAX_SRC];

			/// <summary>
			/// performance
			/// </summary>
			public int perf_runs;
			public int64_t perf_cycles;
			public int64_t perf_time_us;

			public ggml_tensor* view_src;
			public size_t view_offs;

			public IntPtr data;

			public fixed byte name[GGML_MAX_NAME];

			/// <summary>
			/// extra things e.g. for ggml-cuda.cu
			/// </summary>
			public IntPtr extra;

			public fixed byte padding[8];
		};

		public size_t GGML_TENSOR_SIZE => (ulong)sizeof(ggml_tensor);
		public delegate bool ggml_abort_callback(IntPtr data);

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_cplan
		{
			/// <summary>
			/// size of work buffer, calculated by `ggml_graph_plan()`
			/// </summary>
			public size_t work_size;
			/// <summary>
			/// work buffer, to be allocated by caller before calling to `ggml_graph_compute()`
			/// </summary>
			public uint8_t* work_data;

			public int n_threads;

			/// <summary>
			/// abort ggml_graph_compute when true
			/// </summary>
			public ggml_abort_callback? abort_callback;
			public IntPtr abort_callback_data;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_hash_set
		{
			public size_t size;
			public ggml_tensor** keys;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_cgraph
		{
			public int size;
			public int n_nodes;
			public int n_leafs;

			public ggml_tensor** nodes;
			public ggml_tensor** grads;
			public ggml_tensor** leafs;

			public ggml_hash_set? visited_hash_table;

			public ggml_cgraph_eval_order order;

			/// <summary>
			/// performance
			/// </summary>
			public int perf_runs;
			public int64_t perf_cycles;
			public int64_t perf_time_us;
		};

		/// <summary>
		/// scratch buffer
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_scratch
		{
			public size_t offs;
			public size_t size;
			public IntPtr data;
		};

		/// <summary>
		/// memory pool
		/// </summary>
		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_init_params
		{
			/// <summary>
			/// bytes
			/// </summary>
			public size_t mem_size;

			/// <summary>
			/// if NULL, memory will be allocated publicly
			/// </summary>
			public IntPtr mem_buffer;

			/// <summary>
			/// don't allocate memory for the tensor data
			/// </summary>
			public bool no_alloc;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_compute_params
		{
			public ggml_task_type type;

			/// <summary>
			/// ith = thread index, nth = number of threads
			/// </summary>
			public int ith, nth;

			/// <summary>
			/// work buffer for all threads
			/// </summary>
			public size_t wsize;
			public IntPtr wdata;
		};

		public struct ggml_guid_t
		{
			public byte[] Value;

			public ggml_guid_t(byte[] value)
			{
				if (value.Length != 16)
				{
					throw new ArgumentException("GUID must be 16 bytes long.");
				}
				Value = value;
			}
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_context
		{
			public size_t mem_size;
			public IntPtr mem_buffer;
			public bool mem_buffer_owned;
			public bool no_alloc;
			/// <summary>
			/// this is used to save the no_alloc state when using scratch buffers
			/// </summary>
			public bool no_alloc_save;

			public int n_objects;

			public ggml_object* objects_begin;
			public ggml_object* objects_end;

			public ggml_scratch scratch;
			public ggml_scratch scratch_save;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_context_container
		{
			public bool used;

			public ggml_context context;
		};

		[StructLayout(LayoutKind.Sequential, Pack = 8)]
		public struct ggml_opt_params
		{
			public ggml_opt_type type;

			public size_t graph_size;

			public int n_threads;

			// delta-based convergence test
			//
			//   if past == 0 - disabled
			//   if past > 0:
			//     stop if |f(x) - f(x_past)| < delta * max(1, |f(x)|)
			//
			public int past;
			public float delta;

			// maximum number of iterations without improvement
			//
			//   if 0 - disabled
			//   if > 0:
			//     assume convergence if no cost improvement in this number of iterations
			//
			public int max_no_improvement;

			public bool print_forward_graph;
			public bool print_backward_graph;

			public int n_gradient_accumulation;

			/// <summary>
			/// ADAM parameters
			/// </summary>
			public adam _adam;

			/// <summary>
			/// LBFGS parameters
			/// </summary>
			public lbfgs _lbfgs;


			// ADAM parameters
			[StructLayout(LayoutKind.Sequential)]
			public struct adam
			{
				public int n_iter;

				/// <summary>
				/// schedule multiplier (fixed, decay or warmup)
				/// </summary>
				public float sched;

				/// <summary>
				/// weight decay for AdamW, use 0.0f to disable
				/// </summary>
				public float decay;

				/// <summary>
				/// minimum number of tensor dimension to apply weight decay
				/// </summary>
				public int decay_min_ndim;

				/// <summary>
				/// learning rate
				/// </summary>
				public float alpha;
				public float beta1;
				public float beta2;

				/// <summary>
				/// epsilon for numerical stability
				/// </summary>
				public float eps;

				/// <summary>
				/// epsilon for convergence test
				/// </summary>
				public float eps_f;

				/// <summary>
				/// epsilon for convergence test
				/// </summary>
				public float eps_g;

				/// <summary>
				/// gradient clipping
				/// </summary>
				public float gclip;
			}

			[StructLayout(LayoutKind.Sequential)]
			public struct lbfgs
			{
				/// <summary>
				/// number of corrections to approximate the inv. Hessian
				/// </summary>
				public int m;
				public int n_iter;
				public int max_linesearch;

				/// <summary>
				/// convergence tolerance
				/// </summary>
				public float eps;

				/// <summary>
				/// line search tolerance
				/// </summary>
				public float ftol;
				public float wolfe;
				public float min_step;
				public float max_step;

				public ggml_linesearch linesearch;
			}
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_opt_context
		{
			public ggml_context* ctx;
			public ggml_opt_params @params;

			public int iter;

			/// <summary>
			/// number of parameter elements
			/// </summary>
			public int64_t nx;

			public bool just_initialized;

			public float loss_before;
			public float loss_after;
			public adam _adam;
			public lbfgs _lbfgs;

			[StructLayout(LayoutKind.Sequential)]
			public struct adam
			{
				/// <summary>
				/// current gradient
				/// </summary>
				public ggml_tensor* g;

				/// <summary>
				/// first moment
				/// </summary>
				public ggml_tensor* m;

				/// <summary>
				/// second moment
				/// </summary>
				public ggml_tensor* v;

				/// <summary>
				/// past function values
				/// </summary>
				public ggml_tensor* pf;
				public float fx_best;
				public float fx_prev;
				public int n_no_improvement;
			}

			[StructLayout(LayoutKind.Sequential)]
			public struct lbfgs
			{
				/// <summary>
				/// current parameters
				/// </summary>
				public ggml_tensor* x;

				/// <summary>
				/// previous parameters
				/// </summary>
				public ggml_tensor* xp;

				/// <summary>
				/// current gradient
				/// </summary>
				public ggml_tensor* g;

				/// <summary>
				/// previous gradient
				/// </summary>
				public ggml_tensor* gp;

				/// <summary>
				/// search direction
				/// </summary>
				public ggml_tensor* d;

				/// <summary>
				/// past function values
				/// </summary>
				public ggml_tensor* pf;

				/// <summary>
				/// the L-BFGS memory alpha
				/// </summary>
				public ggml_tensor* lmal;

				/// <summary>
				/// the L-BFGS memory ys
				/// </summary>
				public ggml_tensor* lmys;

				/// <summary>
				/// the L-BFGS memory s
				/// </summary>
				public ggml_tensor* lms;

				/// <summary>
				/// the L-BFGS memory y
				/// </summary>
				public ggml_tensor* lmy;
				public float fx_best;
				public float step;
				public int j;
				public int k;
				public int end;
				public int n_no_improvement;
			}
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_header
		{
			public fixed byte magic[4];

			public uint32_t version;

			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n_tensors;

			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n_kv;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_kv
		{
			public gguf_str key;
			public gguf_type type;
			public gguf_value value;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_str
		{
			/// <summary>
			/// GGUFv2
			/// </summary>
			public uint64_t n;
			public IntPtr data;
		};

		[StructLayout(LayoutKind.Explicit, Size = 24)]
		public struct gguf_value
		{
			[FieldOffset(0)] public uint8_t uint8;
			[FieldOffset(0)] public int8_t int8;
			[FieldOffset(0)] public uint16_t uint16;
			[FieldOffset(0)] public int16_t int16;
			[FieldOffset(0)] public uint32_t uint32;
			[FieldOffset(0)] public int32_t int32;
			[FieldOffset(0)] public float float32;
			[FieldOffset(0)] public uint64_t uint64;
			[FieldOffset(0)] public int64_t int64;
			[FieldOffset(0)] public double float64;
			[FieldOffset(0)] public bool bool_;

			[FieldOffset(0)] public gguf_str str;

			[FieldOffset(0)] public arr _arr;

			[StructLayout(LayoutKind.Sequential)]
			public struct arr
			{
				public gguf_type type;

				/// <summary>
				/// GGUFv2
				/// </summary>
				public uint64_t n;
				public IntPtr data;
			}

		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_tensor_info
		{
			public gguf_str name;

			public uint32_t n_dims;
			public fixed uint64_t ne[GGML_MAX_DIMS];

			public ggml_type type;

			/// <summary>
			/// offset from start of `data`, must be a multiple of `ALIGNMENT`
			/// </summary>
			public uint64_t offset;

			// for writing API
			public IntPtr data;
			public size_t size;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_context
		{
			public gguf_header header;

			public gguf_kv* kv;
			public gguf_tensor_info* infos;

			public size_t alignment;

			/// <summary>
			/// offset of `data` from beginning of file
			/// </summary>
			public size_t offset;

			/// <summary>
			/// size of `data` in bytes
			/// </summary>
			public size_t size;

			//uint8_t * padding;
			public IntPtr data;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_init_params
		{
			public bool no_alloc;

			/// <summary>
			/// if not NULL, create a ggml_context and allocate the tensor data in it
			/// </summary>
			public ggml_context** ctx;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct gguf_buf
		{
			public IntPtr data;
			public size_t size;
			public size_t offset;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_bf16_t
		{
			public uint16_t bits;
		}

		#endregion

		#region ggml-backend-impl.h

		public struct ggml_backend_buffer_type_context_t;
		public struct ggml_backend_context_t;

		/// <summary>
		/// buffer
		/// </summary>
		public struct ggml_backend_buffer_context_t;

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer
		{
			public IntPtr iface;
			public IntPtr buft;
			public IntPtr context;
			public size_t size;
			public IntPtr usage;
		};

		public delegate string get_name(ggml_backend_buffer_t buft);
		public delegate ggml_backend_buffer_t alloc_buffer(ggml_backend_buffer_type_t buft, size_t size);
		public delegate size_t get_alignment(ggml_backend_buffer_type_t buft);
		public delegate size_t get_max_size(ggml_backend_buffer_type_t buft);
		public delegate size_t get_alloc_size(ggml_backend_buffer_type_t buft, ggml_tensor* tensor);
		public delegate bool supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend);
		public delegate bool is_host(ggml_backend_buffer_type_t buft);

		public delegate void free_buffer(ggml_backend_buffer_t buffer);
		public delegate IntPtr get_base(ggml_backend_buffer_t buffer);
		public delegate void init_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor);
		public delegate void set_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void get_tensor(ggml_backend_buffer_t buffer, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);

		/// <summary>
		/// // dst is in the buffer, src may be in any buffer
		/// </summary>
		/// <param name="buffer"></param>
		/// <param name="src"></param>
		/// <param name="dst"></param>
		/// <returns></returns>
		public delegate bool cpy_tensor(ggml_backend_buffer_t buffer, ggml_tensor* src, ggml_tensor* dst);
		public delegate void clear(ggml_backend_buffer_t buffer, uint8_t value);

		/// <summary>
		/// // reset any internal state due to tensor initialization, such as tensor extras
		/// </summary>
		/// <param name="buffer"></param>
		public delegate void reset(ggml_backend_buffer_t buffer);

		public delegate void free(ggml_backend_t backend);

		/// <summary>
		/// buffer allocation
		/// </summary>
		/// <param name="backend"></param>
		/// <returns></returns>
		public delegate ggml_backend_buffer_type_t get_default_buffer_type(ggml_backend_t backend);

		/// <summary>
		/// ( ) asynchronous tensor data access
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="tensor"></param>
		/// <param name="data"></param>
		/// <param name="offset"></param>
		/// <param name="size"></param>
		public delegate void set_tensor_async(ggml_backend_t backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void get_tensor_async(ggml_backend_t backend, ggml_tensor* tensor, IntPtr data, size_t offset, size_t size);
		public delegate void cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, ggml_tensor* src, ggml_tensor* dst);

		/// <summary>
		/// ( ) complete all pending operations
		/// </summary>
		/// <param name="backend"></param>
		public delegate void synchronize(ggml_backend_t backend);

		/// <summary>
		/// compute graph with a plan (not used currently)
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="cgraph"></param>
		/// <returns></returns>
		public delegate ggml_backend_graph_plan_t graph_plan_create(ggml_backend_t backend, ggml_cgraph* cgraph);
		public delegate void graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

		/// <summary>
		/// compute graph with a plan
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="plan"></param>
		/// <returns></returns>
		public delegate ggml_status graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan);

		/// <summary>
		/// compute graph without a plan (async)
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="cgraph"></param>
		/// <returns></returns>
		public delegate ggml_status graph_compute(ggml_backend_t backend, ggml_cgraph* cgraph);

		/// <summary>
		/// check if the backend supports an operation
		/// </summary>
		/// <param name="backend"></param>
		/// <param name="op"></param>
		/// <returns></returns>
		public delegate bool supports_op(ggml_backend_t backend, ggml_tensor* op);

		// check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
		// these should be expensive operations with large batch sizes that may benefit from running on this backend
		// even if the weight has to be copied from the CPU temporarily
		public delegate void offload_op(ggml_backend_t backend, ggml_tensor* op);

		/// <summary>
		/// ( ) event synchronization
		/// </summary>
		/// <param name="backend"></param>
		/// <returns></returns>
		public delegate ggml_backend_event_t event_new(ggml_backend_t backend);

		public delegate void event_free(ggml_backend_event_t @event);

		public delegate void event_record(ggml_backend_event_t @event);

		public delegate void event_wait(ggml_backend_t backend, ggml_backend_event_t @event);

		public delegate void event_synchronize(ggml_backend_event_t @event);


		public struct ggml_backend_buffer_type_i
		{
			public get_name get_name;
			public alloc_buffer alloc_buffer;
			public get_alignment get_alignment;
			public get_max_size get_max_size;
			public get_alloc_size get_alloc_size;
			public supports_backend support_backend;
			public is_host is_hose;

		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer_type
		{
			public ggml_backend_buffer_type_i iface;
			public ggml_backend_buffer_type_context_t context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend
		{
			public ggml_guid_t guid;

			ggml_backend_i iface;
			ggml_backend_context_t context;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_buffer_i
		{
			public get_name get_name;
			public free_buffer free_buffer;
			public get_base get_base;
			public init_tensor init_tensor;
			public set_tensor set_tensor;
			public get_tensor get_tensor;
			public cpy_tensor cpy_tensor; // dst is in the buffer, src may be in any buffer
			public clear clear;
			public reset reset; // reset any internal state due to tensor initialization, such as tensor extras
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_i
		{
			public get_name get_name;

			public free free;

			// buffer allocation
			public get_default_buffer_type get_default_buffer_type;

			// ( ) asynchronous tensor data access
			public set_tensor_async set_tensor_async;
			public get_tensor_async get_tensor_async;
			public cpy_tensor_async cpy_tensor_async;

			// ( ) complete all pending operations
			public synchronize synchronize;

			// compute graph with a plan (not used currently)
			public graph_plan_create graph_plan_create;
			public graph_plan_free graph_plan_free;

			// compute graph with a plan
			public graph_plan_compute graph_plan_compute;
			// compute graph without a plan (async)
			public graph_compute graph_compute;

			// check if the backend supports an operation
			public supports_op supports_op;

			// check if the backend wants to run an operation, even if the weights are allocated in a CPU buffer
			// these should be expensive operations with large batch sizes that may benefit from running on this backend
			// even if the weight has to be copied from the CPU temporarily
			public offload_op offload_op;

			// ( ) event synchronization
			public event_new event_new;
			public event_free event_free;

			public event_record event_record;

			public event_wait event_wait;

			public event_synchronize event_synchronize;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_event
		{
			public ggml_backend_t backend;
			public IntPtr context;
		};


		#endregion


		#region ggml-backend.h


		public struct ggml_backend_graph_plan_t;
		public struct ggml_backend_sched;
		public delegate bool ggml_backend_sched_eval_callback(ggml_tensor* t, bool ask, IntPtr user_data);
		public delegate bool ggml_backend_eval_callback(int node_index, ggml_tensor* t1, ggml_tensor* t2, IntPtr user_data);

		/// <summary>
		/// buffer
		/// </summary>
		public enum ggml_backend_buffer_usage
		{
			GGML_BACKEND_BUFFER_USAGE_ANY = 0,
			GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1,
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_backend_graph_copy
		{
			ggml_backend_buffer_t buffer;
			ggml_context* ctx_allocated;
			ggml_context* ctx_unallocated;
			ggml_cgraph* graph;
		};

		#endregion


		#region ggml-alloc.h

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_tallocr
		{
			public ggml_backend_buffer_t buffer;
			public IntPtr @base;
			public size_t alignment;
			public size_t offset;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct ggml_gallocr
		{
			public ggml_backend_buffer_type_t* bufts; // [n_buffers]
			public ggml_backend_buffer_t* buffers; // [n_buffers]
			public ggml_dyn_tallocr** buf_tallocs; // [n_buffers]
			public int n_buffers;

			public ggml_hash_set hash_set;
			public hash_node* hash_values; // [hash_set.size]

			public node_alloc* node_allocs; // [n_nodes]
			public int n_nodes;

			public leaf_alloc* leaf_allocs; // [n_leafs]
			public int n_leafs;
		};

		[StructLayout(LayoutKind.Explicit)]
		public struct ggml_dyn_tallocr
		{
			[FieldOffset(0)] public size_t alignment;
			[FieldOffset(8)] public int n_free_blocks;
			[FieldOffset(16)] public free_block[] free_block;  // free_block's count is  [MAX_FREE_BLOCKS] (256)
			[FieldOffset(4112)] public size_t max_size;
		}

		[StructLayout(LayoutKind.Sequential)]
		public struct free_block
		{
			public size_t offset;
			public size_t size;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct hash_node
		{
			public int n_children;
			public int n_views;
			public int buffer_id;
			public size_t offset; // offset within the buffer
			public bool allocated;
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct leaf_alloc
		{
			public int buffer_id;
			public tensor_alloc leaf;
		};

		[StructLayout(LayoutKind.Explicit)]
		public struct node_alloc
		{
			[FieldOffset(0)] public int buffer_id;
			[FieldOffset(8)] tensor_alloc dst;
			[FieldOffset(24)] tensor_alloc[] src;  //  src' count is  [GGML_MAX_SRC] (10)
		};

		[StructLayout(LayoutKind.Sequential)]
		public struct tensor_alloc
		{
			public size_t offset;
			public size_t size_max; // 0 = pre-allocated, unused, or view
		};

		#endregion



	}
}

