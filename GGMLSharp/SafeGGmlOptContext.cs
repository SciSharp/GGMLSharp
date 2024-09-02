using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static GGMLSharp.InternalStructs;
using static GGMLSharp.Structs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlOptContext : SafeGGmlHandleBase
	{
		private ggml_opt_context* context => (ggml_opt_context*)handle;

		private bool IsInitialized => handle != IntPtr.Zero;

		private void ThrowIfNotInitialized()
		{
			if (!IsInitialized)
			{
				throw new ObjectDisposedException("Not initialized or disposed");
			}
		}

		public long NX => context->nx;
		public float LossBefore => context->loss_before;
		public float LossAfter => context->loss_after;
		public int Iter => context->iter;
		public bool JustInitialized => context->just_initialized;

		//public OptimizerParameters Params =>(OptimizerParameters) context->@params;

	}
}
