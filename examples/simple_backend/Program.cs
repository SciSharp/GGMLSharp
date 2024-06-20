using GGMLSharp;
using System.Runtime.InteropServices;
using static GGMLSharp.Structs;

namespace simple_backend
{
	internal unsafe class Program
	{
		static void Main(string[] args)
		{
			Native.ggml_time_init();

			// initialize data of matrices to perform matrix multiplication
			const int rows_A = 4, cols_A = 2;

			float[] matrix_A = new float[rows_A * cols_A]
			{
				2, 8,
				5, 1,
				4, 2,
				8, 6
			};
			const int rows_B = 3, cols_B = 2;
			/* Transpose([
				10, 9, 5,
				5, 9, 4
			]) 2 rows, 3 cols */
			float[] matrix_B = new float[rows_B * cols_B]
			{
				10, 5,
				9, 9,
				5, 4
			};

			simple_model model = load_model((float*)Marshal.UnsafeAddrOfPinnedArrayElement(matrix_A, 0), (float*)Marshal.UnsafeAddrOfPinnedArrayElement(matrix_B, 0), rows_A, cols_A, rows_B, cols_B);

			// calculate the temporaly memory required to compute
			ggml_gallocr* allocr = Native.ggml_gallocr_new(Native.ggml_backend_get_default_buffer_type(model.backend));

			// create the worst case graph for memory usage estimation
			ggml_cgraph* gf = build_graph(model);
			Native.ggml_gallocr_reserve(allocr, gf);
			ulong mem_size = Native.ggml_gallocr_get_buffer_size(allocr, 0);

			Console.WriteLine($"compute buffer size: {mem_size / 1024.0} KB");

			// perform computation
			ggml_tensor* result = compute(model, allocr);

			// create a array to print result
			float[] out_data = new float[Native.ggml_nelements(result)];

			// bring the data from the backend memory
			Native.ggml_backend_tensor_get(result, Marshal.UnsafeAddrOfPinnedArrayElement(out_data, 0), 0, Native.ggml_nbytes(result));

			// expected result:
			// [ 60.00 110.00 54.00 29.00
			//  55.00 90.00 126.00 28.00
			//  50.00 54.00 42.00 64.00 ]

			Console.WriteLine($"mul mat ({(int)result->ne[0]} x {(int)result->ne[1]}) (transposed result):[");
			for (int j = 0; j < result->ne[1] /* rows */; j++)
			{
				if (j > 0)
				{
					Console.WriteLine();
				}

				for (int i = 0; i < result->ne[0] /* cols */; i++)
				{
					Console.Write($" {out_data[i * result->ne[1] + j]}");
				}
			}
			Console.WriteLine(" ]");

			// release backend memory used for computation
			Native.ggml_gallocr_free(allocr);

			// free memory
			Native.ggml_free(model.ctx);

			// release backend memory and free backend
			Native.ggml_backend_buffer_free(model.buffer);
			Native.ggml_backend_free(model.backend);

		}

		class simple_model
		{
			public ggml_tensor* a;
			public ggml_tensor* b;

			// the backend to perform the computation (CPU, CUDA, METAL)
			public ggml_backend* backend = null;

			// the backend buffer to storage the tensors data of a and b
			public ggml_backend_buffer* buffer;

			// the context to define the tensor information (dimensions, size, memory address)
			public ggml_context* ctx;
		};
		static simple_model load_model(float* a, float* b, int rows_A, int cols_A, int rows_B, int cols_B)
		{
			// initialize the backend
			simple_model model = new simple_model();
			if (Native.ggml_cpu_has_cuda())
			{
				model.backend = Native.ggml_backend_cuda_init(0); // init device 0
			}
			else
			{
				model.backend = Native.ggml_backend_cpu_init();
			}

			if (model.backend == null)
			{
				Console.WriteLine("ggml_backend_cuda_init() failed.");
				Console.WriteLine("we while use ggml_backend_cpu_init() instead.");

				// if there aren't GPU Backends fallback to CPU backend
				model.backend = Native.ggml_backend_cpu_init();
			}

			int num_tensors = 2;
			ggml_init_params @params = new ggml_init_params()
			{
				mem_size = Native.ggml_tensor_overhead() * (ulong)num_tensors,
				mem_buffer = IntPtr.Zero,
				no_alloc = true,
			};

			// create context
			model.ctx = Native.ggml_init(@params);

			// create tensors
			model.a = Native.ggml_new_tensor_2d(model.ctx, ggml_type.GGML_TYPE_F32, cols_A, rows_A);
			model.b = Native.ggml_new_tensor_2d(model.ctx, ggml_type.GGML_TYPE_F32, cols_B, rows_B);

			// create a backend buffer (backend memory) and alloc the tensors from the context
			model.buffer = Native.ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

			// load data from cpu memory to backend buffer
			Native.ggml_backend_tensor_set(model.a, (IntPtr)a, 0, Native.ggml_nbytes(model.a));
			Native.ggml_backend_tensor_set(model.b, (IntPtr)b, 0, Native.ggml_nbytes(model.b));
			return model;
		}

		static ggml_cgraph* build_graph(simple_model model)
		{
			ulong buf_size = Native.ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + Native.ggml_graph_overhead();
			IntPtr mem_buffer = Marshal.AllocHGlobal((int)buf_size);

			ggml_init_params params0 = new ggml_init_params()
			{
				mem_size = buf_size,
				mem_buffer = mem_buffer,// the tensors will be allocated later by ggml_allocr_alloc_graph()
				no_alloc = true,
			};

			// create a temporally context to build the graph
			ggml_context* ctx0 = Native.ggml_init(params0);

			ggml_cgraph* gf = Native.ggml_new_graph(ctx0);

			// result = a*b^T
			ggml_tensor* result = Native.ggml_mul_mat(ctx0, model.a, model.b);

			// build operations nodes
			Native.ggml_build_forward_expand(gf, result);

			// delete the temporally context used to build the graph
			Native.ggml_free(ctx0);
			return gf;
		}

		// compute with backend
		static ggml_tensor* compute(simple_model model, ggml_gallocr* allocr)
		{
			// reset the allocator to free all the memory allocated during the previous inference

			ggml_cgraph* gf = build_graph(model);

			// allocate tensors
			Native.ggml_gallocr_alloc_graph(allocr, gf);

			Native.ggml_backend_graph_compute(model.backend, gf);

			// in this case, the output tensor is the last one in the graph
			return gf->nodes[gf->n_nodes - 1];
		}



	}
}
