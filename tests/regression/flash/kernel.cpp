#include <vx_spawn.h>
#include <vx_tensor.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <cstdio>
#include "common.h"

namespace vt = vortex::tensor;

// Need to be known at compile-time
static constexpr uint32_t HEAD_DIM = 8;
static constexpr uint32_t BLOCK_SIZE_C_MAX = 8;

void flashattention_simt(kernel_arg_t *arg) {
    // Setup buffer arguments
    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto seq_len = arg->seq_len;
    auto block_size_r = arg->block_size_r;
    auto block_size_c = arg->block_size_c;

    // Allocate local memory
    // Must store tile of Q (b_r x d) +  K, V (b_c x d)
    auto local_ptr = __local_mem((block_size_r + 2 * block_size_c) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_K = (float*)local_Q + block_size_r * HEAD_DIM;
    auto local_V = (float*)local_K + block_size_c * HEAD_DIM;

    // Determine global/local row index
    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * HEAD_DIM;
    auto l_row_offset = l_row * HEAD_DIM;

    // Load Q_i from HBM
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Initialize O_i in registers (accumulates l_i * O_i)
    float O_buf[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
      O_buf[col] = 0.0f;

    // Create buffer to store row of S and P
    float sp_buf[BLOCK_SIZE_C_MAX];

    // Initialize m_i, l_i
    float m = -INFINITY;
    float l = 0.0f;

    // Thread's row of Q block
    float* Q_row = local_Q + l_row * HEAD_DIM;

    // Loop over blocks of K and V
    for (uint32_t j = 0; j < seq_len; j += block_size_c) {
        uint32_t block_offset = j * HEAD_DIM;

        // Load K_j and V_j^T
        // block_size_c % block_size_r = 0
        for (uint32_t k = 0; k < block_size_c / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * HEAD_DIM;
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            local_K[offset] = K_ptr[block_offset + offset];
            // Store transpose of V_j
            local_V[col * block_size_c + row] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        // Thread's row of S_ij = Q_i · K_j^T
        // Compute dot product of thread's Q row and each row of K_j
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k)
          sp_buf[k] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < BLOCK_SIZE_C; ++k) {
          // sp_buf[k] = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < HEAD_DIM; ++elem)
            sp_buf[k] += Q_row[elem] * local_K[k * HEAD_DIM + elem];
        }

        // Row max
        float rowmax = sp_buf[0];
        #pragma clang loop unroll(full)
        for (uint32_t k = 1; k < block_size_c; ++k)
          if (sp_buf[k] > rowmax) rowmax = sp_buf[k];

        // Threads row of P_ij
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < block_size_c; ++k)
          sp_buf[k] = expf(sp_buf[k] - rowmax);

        // Row sum
        float rowsum = 0;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < block_size_c; ++k)
          rowsum += sp_buf[k];

        float inv_rowsum = 1.0f / rowsum;

        // Compute new m and l using streaming softmax update
        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight;

        // Update O with normalized probabilities
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          float dot = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < BLOCK_SIZE_C; ++elem)
            dot += (sp_buf[elem] * inv_rowsum) * local_V[k * BLOCK_SIZE_C + elem];
          float old_contrib = (l > 0) ? (old_weight * l * O_buf[k]) : 0.0f;
          float new_contrib = new_weight * dot;
          O_buf[k] = (old_contrib + new_contrib) / new_l;
        }

        // Update m and l for next block
        m = new_m;
        l = new_l;

        __syncthreads();
    }

    // Normalize O and write back to HBM
    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    #pragma clang loop unroll(full)
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
}

void flashattention_tcu(kernel_arg_t *arg) {
    // Placeholder dispatch to SIMT path until TCU path is implemented.
    flashattention_simt(arg);
}

void kernel_body(kernel_arg_t *arg) {
    if (arg->kernel_type == 1)
        flashattention_tcu(arg);
    else
        flashattention_simt(arg);
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
