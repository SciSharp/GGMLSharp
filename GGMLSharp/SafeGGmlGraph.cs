using System;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlGraph : SafeGGmlHandleBase
	{
		public bool IsInitialized => handle != IntPtr.Zero;
		private ggml_cgraph* graph => (ggml_cgraph*)handle;
		public int NodeCount => graph->n_nodes;
		public int LeafCount => graph->n_leafs;

		public long TimeUse => graph->perf_time_us;

		public long Cycles => graph->perf_cycles;

		public int Runs => graph->perf_runs;

		public Structs.GGmlGraphEvalOrder EvalOrder => (Structs.GGmlGraphEvalOrder)graph->order;

		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public SafeGGmlGraph()
		{
			this.handle = IntPtr.Zero;
		}



		public SafeGGmlTensor[] Nodes
		{
			get
			{
				SafeGGmlTensor[] nodes = new SafeGGmlTensor[NodeCount];
				for (int i = 0; i < NodeCount; i++)
				{
					nodes[i] = new SafeGGmlTensor((IntPtr)graph->nodes[i]);
				}
				return nodes;
			}
		}

		public SafeGGmlTensor[] Leafs
		{
			get
			{
				SafeGGmlTensor[] leafs = new SafeGGmlTensor[LeafCount];
				for (int i = 0; i < LeafCount; i++)
				{
					leafs[i] = new SafeGGmlTensor((IntPtr)graph->leafs[i]);
				}
				return leafs;
			}
		}
		public SafeGGmlTensor[] Grads
		{
			get
			{
				if (graph->grads != null)
				{
					SafeGGmlTensor[] grads = new SafeGGmlTensor[NodeCount];
					for (int i = 0; i < NodeCount; i++)
					{
						grads[i] = new SafeGGmlTensor((IntPtr)graph->grads[i]);
					}
					return grads;
				}
				else
				{
					return null;
				}
			}
		}

		public Structs.GGmlStatus ComputeWithGGmlContext(SafeGGmlContext context, int threads)
		{
			ThrowIfNotInitialized();
			return (Structs.GGmlStatus)Native.ggml_graph_compute_with_ctx(context, this, threads);
		}

		public Structs.GGmlStatus Compute(SafeGGmlGraphPlan plan)
		{
			ThrowIfNotInitialized();
			ggml_cplan* p = (ggml_cplan*)plan.DangerousGetHandle();
			return (Structs.GGmlStatus)Native.ggml_graph_compute(this, p);
		}

		public void BuildForwardExpend(SafeGGmlTensor tensor)
		{
			ThrowIfNotInitialized();
			Native.ggml_build_forward_expand(this, tensor);
		}

		public bool GraphAllocate(SafeGGmlGraphAllocr allocr)
		{
			ThrowIfNotInitialized();
			return Native.ggml_gallocr_alloc_graph(allocr, this);
		}

		public Structs.GGmlStatus BackendCompute(SafeGGmlBackend backend)
		{
			ThrowIfNotInitialized();
			return (Structs.GGmlStatus)Native.ggml_backend_graph_compute(backend, this);
		}

		public SafeGGmlTensor GetTensor(string name)
		{
			ThrowIfNotInitialized();
			return Native.ggml_graph_get_tensor(this, name);
		}

		public void Reset()
		{
			ThrowIfNotInitialized();
			Native.ggml_graph_reset(this);
		}

		public void Export(string name)
		{
			Native.ggml_graph_export(this, name);
		}

		public SafeGGmlGraphPlan GetPlan(int threads = -1)
		{
			ggml_cplan p = Native.ggml_graph_plan(this, threads);
			SafeGGmlGraphPlan plan = new SafeGGmlGraphPlan(p);
			return plan;
		}

	}
}

