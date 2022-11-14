#pragma once

#include <torch/extension.h>

void synchronize_cuda();
void read_async_cuda(torch::Tensor src,
                     torch::optional<torch::Tensor> optional_offset,
                     torch::optional<torch::Tensor> optional_count,
                     torch::Tensor index, torch::Tensor dst,
                     torch::Tensor buffer);
void write_async_cuda(torch::Tensor src, torch::Tensor offset,
                      torch::Tensor count, torch::Tensor dst);
