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
        for (uint32_t k = 0; k < block_size_c; ++k)
          sp_buf[k] = 0.0f;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < block_size_c; ++k) {
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

        // Compute new m and l using streaming softmax update
        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;

        // Update O using unnormalized probabilities; final normalization uses l
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          float dot = 0;
          #pragma clang loop unroll(full)
          for (uint32_t elem = 0; elem < block_size_c; ++elem)
            dot += sp_buf[elem] * local_V[k * block_size_c + elem];
          O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
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
    auto seq_len = arg->seq_len;
    auto block_size_r = arg->block_size_r;
    auto block_size_c = arg->block_size_c;

    // Require tile alignment for WMMA fragments
    if (block_size_r != 8 || block_size_c != 8) {
        flashattention_simt(arg);
        return;
    }

    using ctx = vt::wmma_context<BLOCK_SIZE_C_MAX, vt::fp32, vt::fp32>;

    float* Q_ptr = reinterpret_cast<float*>(arg->Q_addr);
    float* K_ptr = reinterpret_cast<float*>(arg->K_addr);
    float* V_ptr = reinterpret_cast<float*>(arg->V_addr);
    float* O_ptr = reinterpret_cast<float*>(arg->O_addr);

    auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
    auto l_row = threadIdx.x;

    auto g_row_offset = g_row * HEAD_DIM;
    auto l_row_offset = l_row * HEAD_DIM;

    // local buffers: Q + K^T + V + Probabilities + O_tile
    auto local_ptr = __local_mem((block_size_r + 2 * block_size_c + block_size_r) * HEAD_DIM * sizeof(float));
    auto local_Q = (float*)local_ptr;
    auto local_Kt = local_Q + block_size_r * HEAD_DIM;
    auto local_V = local_Kt + block_size_c * HEAD_DIM;
    auto local_P = local_V + block_size_c * HEAD_DIM;

    // Load Q tile
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
        local_Q[l_row_offset + col] = Q_ptr[g_row_offset + col];

    // Streaming softmax accumulators
    float O_buf[HEAD_DIM];
    #pragma clang loop unroll(full)
    for (uint32_t col = 0; col < HEAD_DIM; ++col)
      O_buf[col] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    for (uint32_t j = 0; j < seq_len; j += block_size_c) {
        uint32_t block_offset = j * HEAD_DIM;

        // Load K and V blocks
        for (uint32_t k = 0; k < block_size_c / block_size_r; ++k) {
          auto row = k * block_size_r + l_row;
          auto row_offset = row * HEAD_DIM;
          for (uint32_t col = 0; col < HEAD_DIM; ++col) {
            auto offset = row_offset + col;
            float kval = K_ptr[block_offset + offset];
            local_Kt[col * block_size_c + row] = kval; // transpose for K^T
            local_V[row * HEAD_DIM + col] = V_ptr[block_offset + offset];
          }
        }

        __syncthreads();

        // Compute S = Q * K^T using WMMA
        ctx::fragment_a fragA;
        ctx::fragment_b fragB;
        ctx::fragment_acc fragS;
        ctx::fill_fragment(fragS, 0);

        ctx::load_matrix_sync(fragA, local_Q, HEAD_DIM);
        ctx::load_matrix_sync<vt::row_major>(fragB, local_Kt, block_size_c);
        ctx::mma_sync(fragS, fragA, fragB, fragS);

        // Store scores to probability buffer temporarily
        ctx::store_matrix_sync(local_P, fragS, block_size_c);

        float rowmax = local_P[l_row * block_size_c];
        #pragma clang loop unroll(full)
        for (uint32_t k = 1; k < block_size_c; ++k) {
            float v = local_P[l_row * block_size_c + k];
            if (v > rowmax) rowmax = v;
        }

        float rowsum = 0.0f;
        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < block_size_c; ++k) {
            float p = expf(local_P[l_row * block_size_c + k] - rowmax);
            local_P[l_row * block_size_c + k] = p;
            rowsum += p;
        }

        float new_m = (m > rowmax ? m : rowmax);
        float old_weight = expf(m - new_m);
        float new_weight = expf(rowmax - new_m);
        float new_l = old_weight * l + new_weight * rowsum;

        __syncthreads();

        // Compute O_tile contribution = P * V using WMMA (unnormalized probabilities)
        ctx::fragment_a fragP;
        ctx::fragment_b fragV;
        ctx::fragment_acc fragO;
        ctx::fill_fragment(fragO, 0);

        ctx::load_matrix_sync(fragP, local_P, block_size_c);
        ctx::load_matrix_sync<vt::row_major>(fragV, local_V, HEAD_DIM);
        ctx::mma_sync(fragO, fragP, fragV, fragO);

        // Store partial O tile
        ctx::store_matrix_sync(local_P, fragO, HEAD_DIM);

        #pragma clang loop unroll(full)
        for (uint32_t k = 0; k < HEAD_DIM; ++k) {
          float dot = local_P[l_row * HEAD_DIM + k];
          O_buf[k] = old_weight * O_buf[k] + new_weight * dot;
        }

        m = new_m;
        l = new_l;

        __syncthreads();
    }

    float inv_l = (l > 0) ? (1.0f / l) : 0.0f;
    #pragma clang loop unroll(full)
    for (uint32_t k = 0; k < HEAD_DIM; ++k)
      O_ptr[g_row_offset + k] = O_buf[k] * inv_l;
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
