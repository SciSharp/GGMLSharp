using GGMLSharp;
using System;
using static GGMLSharp.Structs;

namespace SimpleBackend
{
	internal class Program
	{
		static void Main(string[] args)
		{
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

			SimpleModel model = LoadModel(matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

			// calculate the temporaly memory required to compute
			SafeGGmlGraphAllocr allocr = new SafeGGmlGraphAllocr(model.backend.GetDefaultBufferType());

			// create the worst case graph for memory usage estimation
			SafeGGmlGraph gf = BuildGraph(model);
			allocr.Reserve(gf);
			ulong mem_size = allocr.GetBufferSize(0);

			Console.WriteLine($"compute buffer size: {mem_size / 1024.0} KB");

			// perform computation
			SafeGGmlTensor result = Compute(model, allocr);

			// bring the data from the backend memory
			//Native.ggml_backend_tensor_get(result, Marshal.UnsafeAddrOfPinnedArrayElement(out_data, 0), 0, Native.ggml_nbytes(result));
			byte[] backendBytes = result.GetBackend();

			// create a array to print result
			float[] out_data = DataConverter.ConvertToFloats(backendBytes);
			// expected result:
			// [ 60.00 110.00 54.00 29.00
			//  55.00 90.00 126.00 28.00
			//  50.00 54.00 42.00 64.00 ]

			Console.WriteLine($"mul mat ({(int)result.Shape[0]} x {(int)result.Shape[1]}) (transposed result):[");
			for (int j = 0; j < result.Shape[1] /* rows */; j++)
			{
				if (j > 0)
				{
					Console.WriteLine();
				}

				for (int i = 0; i < result.Shape[0] /* cols */; i++)
				{
					Console.Write($" {out_data[i * result.Shape[1] + j]}");
				}
			}
			Console.WriteLine(" ]");

			// release backend memory used for computation
			allocr.Free();

			// free memory
			model.ctx.Free();

			// release backend memory and free backend

			//model.buffer.Free();
			//Native.ggml_backend_buffer_free(model.buffer);
			model.backend.Free();
			Console.WriteLine("Done");
			Console.ReadKey();
		}
		class SimpleModel
		{
			public SafeGGmlTensor a;
			public SafeGGmlTensor b;

			// the backend to perform the computation (CPU, CUDA, METAL)
			public SafeGGmlBackend backend = null;

			// the backend buffer to storage the tensors data of a and b
			public SafeGGmlBackendBuffer buffer;

			// the context to define the tensor information (dimensions, size, memory address)
			public SafeGGmlContext ctx;
		};
		static SimpleModel LoadModel(float[] a, float[] b, int rows_A, int cols_A, int rows_B, int cols_B)
		{
			// initialize the backend
			SimpleModel model = new SimpleModel();
			if (SafeGGmlBackend.HasCuda)
			{
				model.backend = SafeGGmlBackend.CudaInit(); // init device 0
			}
			else
			{
				model.backend = SafeGGmlBackend.CpuInit();
			}

			if (model.backend == null)
			{
				Console.WriteLine("ggml_backend_cuda_init() failed.");
				Console.WriteLine("we while use ggml_backend_cpu_init() instead.");

				// if there aren't GPU Backends fallback to CPU backend
				model.backend = SafeGGmlBackend.CpuInit();
			}

			// create context
			model.ctx = new SafeGGmlContext(IntPtr.Zero, NoAllocateMemory: true);

			// create tensors
			model.a = model.ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, cols_A, rows_A);
			model.b = model.ctx.NewTensor2d(Structs.GGmlType.GGML_TYPE_F32, cols_B, rows_B);

			// create a backend buffer (backend memory) and alloc the tensors from the context
			model.buffer = model.ctx.BackendAllocContextTensors(model.backend);

			// load data from cpu memory to backend buffer
			model.a.SetBackend(a);
			model.b.SetBackend(b);
			return model;
		}

		static SafeGGmlGraph BuildGraph(SimpleModel model)
		{
			ulong buf_size = Common.TensorOverheadLength * GGML_DEFAULT_GRAPH_SIZE + Common.GraphOverheadLength;
			byte[] buffer = new byte[buf_size];

			// create a temporally context to build the graph
			SafeGGmlContext ctx0 = new SafeGGmlContext(buffer, true);

			SafeGGmlGraph gf = ctx0.NewGraph();

			// result = a*b^T
			SafeGGmlTensor result = ctx0.MulMat(model.a, model.b);

			// build operations nodes
			gf.BuildForwardExpend(result);

			// delete the temporally context used to build the graph
			ctx0.Free();
			return gf;
		}

		// compute with backend
		static SafeGGmlTensor Compute(SimpleModel model, SafeGGmlGraphAllocr allocr)
		{
			// reset the allocator to free all the memory allocated during the previous inference

			SafeGGmlGraph gf = BuildGraph(model);

			// allocate tensors
			gf.GraphAllocate(allocr);

			gf.BackendCompute(model.backend);

			// in this case, the output tensor is the last one in the graph
			return gf.Nodes[gf.NodeCount - 1];
		}

	}
}
