#include <Python.h>
#include <torch/script.h>
#include <pybind11/pybind11.h>

#ifdef WITH_CUDA
#include "cuda/sync_cuda.h"
#endif

#ifdef _WIN32
PyMODINIT_FUNC PyInit__async(void) { return NULL; }
#endif

void synchronize() {
#ifdef WITH_CUDA
  synchronize_cuda();
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void read_async(torch::Tensor src,
                torch::optional<torch::Tensor> optional_offset,
                torch::optional<torch::Tensor> optional_count,
                torch::Tensor index, torch::Tensor dst, torch::Tensor buffer) {
#ifdef WITH_CUDA
  read_async_cuda(src, optional_offset, optional_count, index, dst, buffer);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

void write_async(torch::Tensor src, torch::Tensor offset, torch::Tensor count,
                 torch::Tensor dst) {
#ifdef WITH_CUDA
  write_async_cuda(src, offset, count, dst);
#else
  AT_ERROR("Not compiled with CUDA support");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("synchronize", &synchronize, "synchronize");
  m.def("read_async", &read_async, "read_async");
  m.def("write_async", &write_async, "write_async");
}