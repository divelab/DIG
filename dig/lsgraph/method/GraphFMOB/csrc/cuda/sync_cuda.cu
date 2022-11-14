#include "sync_cuda.h"

#include <ATen/cuda/CUDAContext.h>

#include "../thread.h"

Thread &getThread() {
  static Thread thread;
  return thread;
}

void synchronize_cuda() { getThread().synchronize(); }

void read_async_cuda(torch::Tensor src,
                     torch::optional<torch::Tensor> optional_offset,
                     torch::optional<torch::Tensor> optional_count,
                     torch::Tensor index, torch::Tensor dst,
                     torch::Tensor buffer) {

  AT_ASSERTM(!src.is_cuda(), "Source tensor must be a CPU tensor");
  AT_ASSERTM(!index.is_cuda(), "Index tensor must be a CPU tensor");
  AT_ASSERTM(dst.is_cuda(), "Target tensor must be a CUDA tensor");
  AT_ASSERTM(!buffer.is_cuda(), "Buffer tensor must be a CPU tensor");

  AT_ASSERTM(buffer.is_pinned(), "Buffer tensor must be pinned");

  AT_ASSERTM(src.is_contiguous(), "Source tensor must be contiguous");
  AT_ASSERTM(dst.is_contiguous(), "Target tensor must be contiguous");
  AT_ASSERTM(buffer.is_contiguous(), "Buffer tensor must be contiguous");

  AT_ASSERTM(index.dim() == 1, "Index tensor must be one-dimensional");

  int64_t numel = 0;
  if (optional_offset.has_value()) {
    AT_ASSERTM(src.is_pinned(), "Source tensor must be pinned");
    auto offset = optional_offset.value();
    AT_ASSERTM(!offset.is_cuda(), "Offset tensor must be a CPU tensor");
    AT_ASSERTM(offset.is_contiguous(), "Offset tensor must be contiguous");
    AT_ASSERTM(offset.dim() == 1, "Offset tensor must be one-dimensional");
    AT_ASSERTM(optional_count.has_value(), "Count tensor is undefined");
    auto count = optional_count.value();
    AT_ASSERTM(!count.is_cuda(), "Count tensor must be a CPU tensor");
    AT_ASSERTM(count.is_contiguous(), "Count tensor must be contiguous");
    AT_ASSERTM(count.dim() == 1, "Count tensor must be one-dimensional");
    AT_ASSERTM(offset.numel() == count.numel(), "Size mismatch");
    numel = count.sum().data_ptr<int64_t>()[0];
  }

  AT_ASSERTM(numel + index.numel() <= buffer.size(0),
             "Buffer tensor size too small");
  AT_ASSERTM(numel + index.numel() <= dst.size(0),
             "Target tensor size too small");

  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(src.get_device()),
             "Asynchronous read requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "read_async", [&] {
    getThread().run([=] {
      int64_t size = src.numel() / src.size(0);
      auto src_data = src.data_ptr<scalar_t>();
      auto dst_data = dst.data_ptr<scalar_t>();

      if (optional_offset.has_value()) {
        auto offset = optional_offset.value();
        auto count = optional_count.value();
        auto offset_data = offset.data_ptr<int64_t>();
        auto count_data = count.data_ptr<int64_t>();
        int64_t src_offset, dst_offset = 0, c;
        for (int64_t i = 0; i < offset.numel(); i++) {
          src_offset = offset_data[i], c = count_data[i];
          AT_ASSERTM(src_offset + c <= src.size(0), "Invalid index");
          AT_ASSERTM(dst_offset + c <= dst.size(0), "Invalid index");
          cudaMemcpyAsync(
              dst_data + (dst_offset * size), src_data + (src_offset * size),
              c * size * sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
          dst_offset += c;
        }
      }

      auto _buffer = buffer.narrow(0, 0, index.numel()); // convert to non-const
      torch::index_select_out(_buffer, src, 0, index);
      int64_t dim = src.numel() / src.size(0);
      cudaMemcpyAsync(dst_data + numel * size, buffer.data_ptr<scalar_t>(),
                      index.numel() * dim * sizeof(scalar_t),
                      cudaMemcpyHostToDevice, stream);
    });
  });
}

void write_async_cuda(torch::Tensor src, torch::Tensor offset,
                      torch::Tensor count, torch::Tensor dst) {
  AT_ASSERTM(src.is_cuda(), "Source tensor must be a CUDA tensor");
  AT_ASSERTM(!offset.is_cuda(), "Offset tensor must be a CPU tensor");
  AT_ASSERTM(!count.is_cuda(), "Count tensor must be a CPU tensor");
  AT_ASSERTM(!dst.is_cuda(), "Target tensor must be a CPU tensor");

  AT_ASSERTM(dst.is_pinned(), "Target tensor must be pinned");

  AT_ASSERTM(src.is_contiguous(), "Index tensor must be contiguous");
  AT_ASSERTM(offset.is_contiguous(), "Offset tensor must be contiguous");
  AT_ASSERTM(count.is_contiguous(), "Count tensor must be contiguous");
  AT_ASSERTM(dst.is_contiguous(), "Target tensor must be contiguous");

  AT_ASSERTM(offset.dim() == 1, "Offset tensor must be one-dimensional");
  AT_ASSERTM(count.dim() == 1, "Count tensor must be one-dimensional");
  AT_ASSERTM(offset.numel() == count.numel(), "Size mismatch");

  auto stream = at::cuda::getCurrentCUDAStream(src.get_device());
  AT_ASSERTM(stream != at::cuda::getDefaultCUDAStream(src.get_device()),
             "Asynchronous write requires a non-default CUDA stream");

  AT_DISPATCH_ALL_TYPES(src.scalar_type(), "write_async", [&] {
    int64_t size = src.numel() / src.size(0);
    auto src_data = src.data_ptr<scalar_t>();
    auto offset_data = offset.data_ptr<int64_t>();
    auto count_data = count.data_ptr<int64_t>();
    auto dst_data = dst.data_ptr<scalar_t>();
    int64_t src_offset = 0, dst_offset, c;
    for (int64_t i = 0; i < offset.numel(); i++) {
      dst_offset = offset_data[i], c = count_data[i];
      AT_ASSERTM(src_offset + c <= src.size(0), "Invalid index");
      AT_ASSERTM(dst_offset + c <= dst.size(0), "Invalid index");
      cudaMemcpyAsync(
          dst_data + (dst_offset * size), src_data + (src_offset * size),
          c * size * sizeof(scalar_t), cudaMemcpyDeviceToHost, stream);
      src_offset += c;
    }
  });
}
