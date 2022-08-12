#include "relabel_cpu.h"

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>,
           torch::Tensor>
relabel_one_hop_cpu(torch::Tensor rowptr, torch::Tensor col,
                    torch::optional<torch::Tensor> optional_value,
                    torch::Tensor idx, bool bipartite) {

  AT_ASSERTM(!rowptr.is_cuda(), "Rowptr tensor must be a CPU tensor");
  AT_ASSERTM(!col.is_cuda(), "Col tensor must be a CPU tensor");
  if (optional_value.has_value()) {
    auto value = optional_value.value();
    AT_ASSERTM(!value.is_cuda(), "Value tensor must be a CPU tensor");
    AT_ASSERTM(value.dim() == 1, "Value tensor must be one-dimensional");
  }
  AT_ASSERTM(!idx.is_cuda(), "Index tensor must be a CPU tensor");

  auto rowptr_data = rowptr.data_ptr<int64_t>();
  auto col_data = col.data_ptr<int64_t>();
  auto idx_data = idx.data_ptr<int64_t>();

  std::vector<int64_t> n_ids;
  std::unordered_map<int64_t, int64_t> n_id_map;
  std::unordered_map<int64_t, int64_t>::iterator it;

  auto out_rowptr = torch::empty(idx.numel() + 1, rowptr.options());
  auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();

  out_rowptr_data[0] = 0;
  int64_t v, w, c, row_start, row_end, offset = 0;
  for (int64_t i = 0; i < idx.numel(); i++) {
    v = idx_data[i];
    n_id_map[v] = i;
    offset += rowptr_data[v + 1] - rowptr_data[v];
    out_rowptr_data[i + 1] = offset;
  }

  auto out_col = torch::empty(offset, col.options());
  auto out_col_data = out_col.data_ptr<int64_t>();

  torch::optional<torch::Tensor> out_value = torch::nullopt;
  if (optional_value.has_value()) {
    out_value = torch::empty(offset, optional_value.value().options());

    AT_DISPATCH_ALL_TYPES(optional_value.value().scalar_type(), "relabel", [&] {
      auto value_data = optional_value.value().data_ptr<scalar_t>();
      auto out_value_data = out_value.value().data_ptr<scalar_t>();

      offset = 0;
      for (int64_t i = 0; i < idx.numel(); i++) {
        v = idx_data[i];
        row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

        for (int64_t j = row_start; j < row_end; j++) {
          w = col_data[j];
          it = n_id_map.find(w);
          if (it == n_id_map.end()) {
            c = idx.numel() + n_ids.size();
            n_id_map[w] = c;
            n_ids.push_back(w);
            out_col_data[offset] = c;
          } else {
            out_col_data[offset] = it->second;
          }
          out_value_data[offset] = value_data[j];
          offset++;
        }
      }
    });

  } else {
    offset = 0;
    for (int64_t i = 0; i < idx.numel(); i++) {
      v = idx_data[i];
      row_start = rowptr_data[v], row_end = rowptr_data[v + 1];

      for (int64_t j = row_start; j < row_end; j++) {
        w = col_data[j];
        it = n_id_map.find(w);
        if (it == n_id_map.end()) {
          c = idx.numel() + n_ids.size();
          n_id_map[w] = c;
          n_ids.push_back(w);
          out_col_data[offset] = c;
        } else {
          out_col_data[offset] = it->second;
        }
        offset++;
      }
    }
  }

  if (!bipartite)
    out_rowptr = torch::cat(
        {out_rowptr, torch::full({(int64_t)n_ids.size()}, out_col.numel(),
                                 rowptr.options())});

  idx = torch::cat({idx, torch::from_blob(n_ids.data(), {(int64_t)n_ids.size()},
                                          idx.options())});

  return std::make_tuple(out_rowptr, out_col, out_value, idx);
}
