add_library(libcua INTERFACE)
target_link_libraries(libcua INTERFACE ${CUDA_LIBRARIES})
target_include_directories(libcua INTERFACE ${CMAKE_CURRENT_LIST_DIR})
