#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Keep the sources list in setup.py aligned with this path.
// Launcher implemented in kernels/FlashAttentionKernel.cu
extern "C" void launch_flash_attention_naive(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B,
    int H,
    int N,
    int D,
    cudaStream_t stream
);

static torch::Tensor flash_attention_naive_forward(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "Q must be float32");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "K must be float32");
    TORCH_CHECK(V.dtype() == torch::kFloat32, "V must be float32");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must have shape [B, H, N, D]");
    TORCH_CHECK(K.sizes() == Q.sizes(), "K must match Q shape");
    TORCH_CHECK(V.sizes() == Q.sizes(), "V must match Q shape");

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int N = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));

    const c10::cuda::CUDAGuard device_guard(Q.device());
    auto O = torch::zeros({B, H, N, D}, Q.options());

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    launch_flash_attention_naive(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        B,
        H,
        N,
        D,
        stream
    );

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attention_naive", &flash_attention_naive_forward,
          "FlashAttention naive (CUDA)");
}
