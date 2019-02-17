add_library(cuda_array INTERFACE)
target_link_libraries(cuda_array INTERFACE ${CUDA_LIBRARIES})
target_include_directories(cuda_array INTERFACE ${CMAKE_CURRENT_LIST_DIR})
