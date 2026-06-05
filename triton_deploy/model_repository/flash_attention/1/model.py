import os

import torch
from torch.utils.cpp_extension import load

import triton_python_backend_utils as pb_utils


def load_extension():
    # Compile the extension on first load and cache the build artifacts.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    build_dir = os.environ.get("FLASH_ATTN_BUILD_DIR", "/tmp/flash_attn_ext")
    sources = [
        os.path.join(repo_root, "csrc", "flash_attn_ext.cpp"),
        os.path.join(repo_root, "kernels", "FlashAttentionKernel.cu"),
    ]
    return load(
        name="flash_attn_ext",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        build_directory=build_dir,
        verbose=False,
    )


class TritonPythonModel:
    def initialize(self, args):
        # Load the CUDA extension once per model instance.
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to run the FlashAttention backend.")
        self.ext = load_extension()

    def execute(self, requests):
        responses = []

        for request in requests:
            q = pb_utils.get_input_tensor_by_name(request, "Q").as_numpy()
            k = pb_utils.get_input_tensor_by_name(request, "K").as_numpy()
            v = pb_utils.get_input_tensor_by_name(request, "V").as_numpy()

            # Basic shape checks to provide clear errors to clients.
            if q.shape != k.shape or q.shape != v.shape:
                error = pb_utils.TritonError("Q, K, and V must have the same shape.")
                responses.append(pb_utils.InferenceResponse(error=error))
                continue
            if q.dtype != k.dtype or q.dtype != v.dtype:
                error = pb_utils.TritonError("Q, K, and V must have the same dtype.")
                responses.append(pb_utils.InferenceResponse(error=error))
                continue

            # Move inputs to GPU and run the custom kernel.
            q_t = torch.from_numpy(q).cuda().contiguous()
            k_t = torch.from_numpy(k).cuda().contiguous()
            v_t = torch.from_numpy(v).cuda().contiguous()

            with torch.no_grad():
                out = self.ext.flash_attention_naive(q_t, k_t, v_t)

            out_np = out.detach().cpu().numpy()
            out_tensor = pb_utils.Tensor("O", out_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
