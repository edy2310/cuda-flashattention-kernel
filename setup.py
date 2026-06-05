from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Simple build script to compile the CUDA extension once.
# Usage: python setup.py build_ext --inplace


def get_extensions():
    return [
        CUDAExtension(
            name="flash_attn_ext",
            sources=[
                "csrc/flash_attn_ext.cpp",
                "kernels/FlashAttentionKernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ]


setup(
    name="flash_attn_ext",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
