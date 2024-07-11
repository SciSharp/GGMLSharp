using System;
using System.Runtime.InteropServices;

namespace GGMLSharp
{
	public class SafeGGmlHandleBase : SafeHandle, IDisposable
	{
		public SafeGGmlHandleBase(IntPtr handle, bool ownsHandle) : base(handle, ownsHandle)
		{
			SetHandle(handle);
		}

		public SafeGGmlHandleBase() : base(IntPtr.Zero, ownsHandle: true)
		{

		}

		public override bool IsInvalid => handle == IntPtr.Zero;

		protected override bool ReleaseHandle()
		{
			if (handle != IntPtr.Zero)
			{
				handle = IntPtr.Zero;
			}
			return true;
		}
	}
}
