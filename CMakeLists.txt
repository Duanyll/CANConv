cmake_minimum_required (VERSION 3.21)

project(CANConv)

add_subdirectory(canconv/layers/pwac)
add_subdirectory(kmcuda/src)

# Copy kmcuda shared library to canconv/layers/kmeans/libKMCUDA.so
add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/canconv/layers/kmeans/libKMCUDA.so
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:KMCUDA> ${CMAKE_SOURCE_DIR}/canconv/layers/kmeans/libKMCUDA.so
  DEPENDS KMCUDA
)
add_custom_target(copy_kmcuda ALL DEPENDS ${CMAKE_SOURCE_DIR}/canconv/layers/kmeans/libKMCUDA.so)