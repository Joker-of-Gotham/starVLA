include_guard(GLOBAL)

# Function uses the CUDA runtime API to query the compute capability of the device, so if a user
# doesn't pass any architecture options to CMake we only build the current architecture

# Adapted from https://github.com/rapidsai/rapids-cmake/blob/branch-24.04/rapids-cmake/cuda/detail/detect_architectures.cmake

function(cuda_detect_architectures_from_gpu possible_archs_var gpu_archs)

  set(__gpu_archs ${${possible_archs_var}})

  set(eval_file eval_gpu_archs.cu)
  set(eval_exe eval_gpu_archs)
  set(error_file eval_gpu_archs.stderr.log)

  if(NOT DEFINED CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "No CUDA compiler specified, unable to determine machine's GPUs.")
  endif()

  if(NOT EXISTS "${eval_exe}")
    file(WRITE ${eval_file}
         "
#include <cstdio>
#include <set>
#include <string>
using namespace std;
int main(int argc, char** argv) {
  set<string> archs;
  int nDevices;
  if((cudaGetDeviceCount(&nDevices) == cudaSuccess) && (nDevices > 0)) {
    for(int dev=0;dev<nDevices;++dev) {
      char buff[32];
      cudaDeviceProp prop;
      if(cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;
      sprintf(buff, \"%d%d\", prop.major, prop.minor);
      archs.insert(buff);
    }
  }
  if(archs.empty()) {
    printf(\"${__gpu_archs}\");
  } else {
    bool first = true;
    for(const auto& arch : archs) {
      printf(first? \"%s\" : \";%s\", arch.c_str());
      first = false;
    }
  }
  printf(\"\\n\");
  return 0;
  }
  ")
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -std=c++11 -o "${eval_exe}" "${eval_file}"
                    ERROR_FILE "${error_file}")
  endif()

  if(EXISTS "${eval_exe}")
    execute_process(COMMAND "./${eval_exe}" OUTPUT_VARIABLE __gpu_archs
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_FILE "${error_file}")
    message(STATUS "Auto detection of gpu-archs: ${__gpu_archs}")
  else()
    message(STATUS "Failed auto detection of gpu-archs. Falling back to using ${__gpu_archs}.")
  endif()
  # remove the build artifacts
  file(REMOVE "${eval_file}" "${eval_exe}" "${error_file}")
  set(${gpu_archs} ${__gpu_archs} PARENT_SCOPE)

endfunction()

# Function to detect CUDA architecture without requiring a GPU
function(cuda_detect_architectures_from_nvcc output_variable)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        OUTPUT_VARIABLE NVCC_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" NVCC_VERSION "${NVCC_OUT}")
    set(NVCC_MAJOR ${CMAKE_MATCH_1})
    set(NVCC_MINOR ${CMAKE_MATCH_2})

    # Base architecture list (Turing and newer)
    set(ARCH_LIST "75;80;86;89;90;100")

    # Add older architectures only for CUDA < 13.0
    if(NVCC_MAJOR LESS 13)
        list(PREPEND ARCH_LIST "52;60;70")  # Maxwell, Pascal, Volta
        message(STATUS "Including SM52/SM60/SM70 support for CUDA ${NVCC_MAJOR}.${NVCC_MINOR}")
    endif()

    set(${output_variable} "${ARCH_LIST}" PARENT_SCOPE)
    message(STATUS "Final architecture list: ${ARCH_LIST}")
endfunction()
