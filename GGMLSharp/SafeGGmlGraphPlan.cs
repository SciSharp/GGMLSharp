using System;
using System.Runtime.InteropServices;
using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
	public unsafe class SafeGGmlGraphPlan : SafeGGmlHandleBase
	{
		private ggml_cplan* plan => (ggml_cplan*)handle;

		public SafeGGmlGraphPlan()
		{
			this.handle = IntPtr.Zero;
		}

		internal SafeGGmlGraphPlan(ggml_cplan plan)
		{
			this.handle = (IntPtr)(&plan);
		}

		internal SafeGGmlGraphPlan(ggml_cplan* plan)
		{
			this.handle = (IntPtr)plan;
		}

		public int Threads => plan->n_threads;
		public ulong WorkSize => plan->work_size;

		public byte[] WorkData
		{
			get
			{
				byte[] bytes = new byte[WorkSize];
				Marshal.Copy((IntPtr)(plan->work_data), bytes, 0, bytes.Length);
				return bytes;
			}
			set
			{
				plan->work_data = Marshal.AllocHGlobal(value.Length);
				Marshal.Copy(value, 0, (IntPtr)plan->work_data, value.Length);
			}
		}

		public IntPtr AbortCallBackData => (IntPtr)plan->abort_callback_data;
		//public ggml_abort_callback AbortCallBack => plan.abort_callback;
	}
}
