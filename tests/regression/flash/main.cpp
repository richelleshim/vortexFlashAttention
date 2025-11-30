#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include <cmath>
#include <algorithm>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
private:
  union Float_t { float f; int i; };
public:
  static const char* type_str() {
    return "float";
  }
  // Modified generate to larger range [-5, 5]
  static float generate() {
      return 10.0f * (float(rand()) / RAND_MAX) - 5.0f;  \
  }
  // Modified compare to allow for greater tolerance
  static bool compare(float a, float b, int index, int errors) {
    const float atol = 1e-5f;
    const float rtol = 1e-5f;
    auto diff = std::fabs(a - b);
    auto limit = atol + rtol * std::fabs(b);
    if (diff > limit) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }
};

static void attention_cpu(float* out, const float* Q, const float* K, const float* V, uint32_t N, uint32_t d) {
  std::vector<float> scores(N);
  std::vector<float> probs(N);
  for (uint32_t i = 0; i < N; ++i) {
    // Compute row of scores
    for (uint32_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (uint32_t k = 0; k < d; ++k) 
        sum += Q[i * d + k] * K[j * d + k];
      scores[j] = sum;
    }

    // Compute softmax of row
    float max = scores[0];
    for (uint32_t j = 1; j < N; ++j)
      max = std::max(max, scores[j]);
    float exp_sum = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
      float exp = std::exp(scores[j] - max);
      probs[j] = exp;
      exp_sum += exp;
    }
    for (uint32_t j = 0; j < N; ++j)
      probs[j] /= exp_sum;

    // Compute row of O
    for (uint32_t k = 0; k < d; ++k) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < N; ++j) 
        sum += probs[j] * V[j * d + k];
      out[i * d + k] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
const char* kernel_kind = "simt";
uint32_t N = 64;
uint32_t d = 8;

vx_device_h device = nullptr;
vx_buffer_h Q_buffer = nullptr;
vx_buffer_h K_buffer = nullptr;
vx_buffer_h V_buffer = nullptr;
vx_buffer_h O_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [--kernel simt|tcu] [-n:sequence_len] [-d:head_dim] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  // parse long options manually
  for (int idx = 1; idx < argc; ++idx) {
    if (0 == strncmp(argv[idx], "--kernel=", 9)) {
      kernel_kind = argv[idx] + 9;
    }
  }

  while ((c = getopt(argc, argv, "n:d:k:h")) != -1) {
    switch (c) {
    case 'n':
      // N = atoi(optarg);
      break;
    case 'd':
      // d = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(Q_buffer);
    vx_mem_free(K_buffer);
    vx_mem_free(V_buffer);
    vx_mem_free(O_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // if ((size / tile_size) * tile_size != size) {
  //   printf("Error: matrix size %d must be a multiple of tile size %d\n", size, tile_size);
  //   return -1;
  // }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));
  
  // // get size M of SRAM
  // uint64_t local_mem_size;
  // vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size);
  // std::cout << "local_mem_size=" << local_mem_size << " bytes" << std::endl;
  // uint32_t M = local_mem_size / sizeof(TYPE);

  // calculate block sizes
  // uint32_t block_size_c = std::min(static_cast<uint32_t>(std::ceil(M / (4 * d))), N);
  // uint32_t block_size_r = std::min(block_size_c, d);
  uint32_t block_size_c = (0 == strcmp(kernel_kind, "tcu")) ? 8u : 4u;
  uint32_t block_size_r = block_size_c;

  uint32_t size = N * d;
  uint32_t buf_size = size * sizeof(TYPE);
  uint32_t group_size = block_size_r;

  uint32_t local_mem = (block_size_r + 2 * block_size_c) * d * sizeof(TYPE);

  // check work group occupancy
  uint32_t max_localmem;
  RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes" << std::endl;
  RT_CHECK(max_localmem < local_mem);

  std::cout << "data type: " << Comparator<float>::type_str() << std::endl;
  std::cout << "sequence length: " << N << std::endl;
  std::cout << "head dimension: " << d << std::endl;
  std::cout << "local memory: " << local_mem << " bytes" << std::endl;
  std::cout << "kernel kind: " << kernel_kind << std::endl;

  // Set kernel args
  kernel_arg.grid_dim[0] = N / block_size_r;
  kernel_arg.block_dim[0] = block_size_r;
  kernel_arg.seq_len = N;
  kernel_arg.head_dim = d;
  kernel_arg.block_size_r = block_size_r;
  kernel_arg.block_size_c = block_size_c;
  kernel_arg.kernel_type = (0 == strcmp(kernel_kind, "tcu")) ? 1u : 0u;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &Q_buffer));
  RT_CHECK(vx_mem_address(Q_buffer, &kernel_arg.Q_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &K_buffer));
  RT_CHECK(vx_mem_address(K_buffer, &kernel_arg.K_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &V_buffer));
  RT_CHECK(vx_mem_address(V_buffer, &kernel_arg.V_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &O_buffer));
  RT_CHECK(vx_mem_address(O_buffer, &kernel_arg.O_addr));

  std::cout << "Q_addr=0x" << std::hex << kernel_arg.Q_addr << std::endl;
  std::cout << "K_addr=0x" << std::hex << kernel_arg.K_addr << std::endl;
  std::cout << "V_addr=0x" << std::hex << kernel_arg.V_addr << std::endl;
  std::cout << "O_addr=0x" << std::hex << kernel_arg.O_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<float> h_Q(size);
  std::vector<float> h_K(size);
  std::vector<float> h_V(size);
  std::vector<float> h_O(size);

  // generate source data
  for (uint32_t i = 0; i < size; ++i) {
    h_Q[i] = Comparator<float>::generate();
    h_K[i] = Comparator<float>::generate();
    h_V[i] = Comparator<float>::generate();
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_copy_to_dev(Q_buffer, h_Q.data(), 0, buf_size));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_copy_to_dev(K_buffer, h_K.data(), 0, buf_size));

  // upload source buffer2
  std::cout << "upload source buffer2" << std::endl;
  RT_CHECK(vx_copy_to_dev(V_buffer, h_V.data(), 0, buf_size));
  
  // upload source buffer3
  std::cout << "upload source buffer3" << std::endl;
  RT_CHECK(vx_copy_to_dev(O_buffer, h_O.data(), 0, buf_size));

  // Upload kernel binary
  std::cout << "Upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_O.data(), O_buffer, 0, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<TYPE> h_ref(size);
    attention_cpu(h_ref.data(), h_Q.data(), h_K.data(), h_V.data(), N, d);

    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_O[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}