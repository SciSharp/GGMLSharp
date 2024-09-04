using System;

using static GGMLSharp.InternalStructs;

namespace GGMLSharp
{
    public unsafe class SafeGGmlBackend : SafeGGmlHandleBase
    {
        private ggml_backend* ggml_backend => (ggml_backend*)handle;

        public SafeGGmlBackend()
        {
            this.handle = IntPtr.Zero;
        }

        public SafeGGmlBackendBufferType GetDefaultBufferType()
        {
            return Native.ggml_backend_get_default_buffer_type(this);
        }

        public static SafeGGmlBackend CpuInit()
        {
            return Native.ggml_backend_cpu_init();
        }

        public static SafeGGmlBackend CudaInit(int index = 0)
        {
            if (!HasCuda)
            {
                throw new NotSupportedException("Cuda Not Support");
            }
            return Native.ggml_backend_cuda_init(index);
        }

        public static SafeGGmlBackend VulkanInit(int index = 0)
        {
            if (!HasVulkan)
            {
                throw new NotSupportedException("Vulkan Not Support");
            }
            return Native.ggml_backend_vk_init(index);
        }

        public static bool HasCuda => Native.ggml_cpu_has_cuda();

        public static bool HasVulkan => Native.ggml_cpu_has_vulkan();

        public void Free()
        {
            Native.ggml_backend_free(this);
        }

    }
}
