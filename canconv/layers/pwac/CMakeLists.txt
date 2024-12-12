cmake_minimum_required (VERSION 3.21)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 17)
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

project ("PWAC" LANGUAGES CXX CUDA)

if (MSVC)
  # suppress C4624
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4624")
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message("TORCH_PYTHON_LIBRARY: ${TORCH_PYTHON_LIBRARY}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message("CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message("Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")

include_directories(${Python3_INCLUDE_DIRS})

add_library(pwacnative SHARED 
  "main.cc" 
  "naive_conv.cu" 
  "im2col.cc" 
  "utils.cc" 
  "matmul_conv.cc"
  "matmul_attn.cc"
  "dispatch_indice.cc"
  "permute_kernel.cu"
  "permute.cc"
)
target_link_libraries(pwacnative 
  ${Python3_LIBRARIES}
  ${TORCH_LIBRARIES}
  ${TORCH_PYTHON_LIBRARY}
)
target_compile_definitions(pwacnative PRIVATE TORCH_EXTENSION_NAME=pwacnative)

# Copy output binary to source directory
if (WIN32)
  add_custom_command(TARGET pwacnative POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pwacnative> ${CMAKE_CURRENT_LIST_DIR}/pwacnative.pyd
  )
else()
  add_custom_command(TARGET pwacnative POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:pwacnative> ${CMAKE_CURRENT_LIST_DIR}/pwacnative.so
  )
endif()