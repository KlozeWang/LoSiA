// bfloat16 version

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <chrono>

template <typename hash_index_type, typename key_type>
__global__ void index_select_kernel_dim2(hash_index_type* __restrict__ out,
                                    const hash_index_type* __restrict__ input,
                                    const int64_t* __restrict__ indices_dim1,
                                    const int64_t* __restrict__ indices_dim2,
                                    int64_t indices_count_dim1,
                                    int64_t indices_count_dim2,
                                    int64_t M) {
  uint64_t col = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < indices_count_dim1 && col < indices_count_dim2) {
    auto current_index_x = indices_dim1[row];
    auto current_index_y = indices_dim2[col];
    out[indices_count_dim2 * row + col] = input[current_index_x * M + current_index_y];
  }
}

template <>
__global__ void index_select_kernel_dim2<at::BFloat16, int64_t>(
    at::BFloat16* __restrict__ out,
    const at::BFloat16* __restrict__ input,
    const int64_t* __restrict__ indices_dim1,
    const int64_t* __restrict__ indices_dim2,
    int64_t indices_count_dim1,
    int64_t indices_count_dim2,
    int64_t M) {
  uint64_t col = blockDim.x * blockIdx.x + threadIdx.x;
  uint64_t row = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < indices_count_dim1 && col < indices_count_dim2) {
    auto current_index_x = indices_dim1[row];
    auto current_index_y = indices_dim2[col];
    out[indices_count_dim2 * row + col] = input[current_index_x * M + current_index_y];
  }
}

torch::Tensor index_select_dim2(torch::Tensor keys, torch::Tensor indices_dim1, torch::Tensor indices_dim2, uint blocks) {
  auto key_tensor_options = torch::TensorOptions()
                            .dtype(keys.dtype())
                            .device(torch::kCUDA);
  auto out = torch::empty({indices_dim1.size(0), indices_dim2.size(0)}, key_tensor_options);

  dim3 block_size(blocks, blocks);
  dim3 grid_size((indices_dim2.size(0) + block_size.x - 1) / block_size.x,
                 (indices_dim1.size(0) + block_size.y - 1) / block_size.y);

  AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::BFloat16, at::ScalarType::Half, keys.scalar_type(), "index_select_dim2", [&] {
    index_select_kernel_dim2<scalar_t, int64_t><<<grid_size, block_size>>>(
        out.data_ptr<scalar_t>(), keys.data_ptr<scalar_t>(),
        indices_dim1.data_ptr<int64_t>(),
        indices_dim2.data_ptr<int64_t>(),
        indices_dim1.size(0),
        indices_dim2.size(0),
        keys.size(1));
  });

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("index_select_dim2", &index_select_dim2, "Index select for 2D tensors");
}