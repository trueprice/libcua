# CudaArray: header-only library for interfacing with CUDA array-type objects
# Author: True Price <jtprice at cs.unc.edu>
#
# BSD License
# Copyright (C) 2017-2019  The University of North Carolina at Chapel Hill
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of the original author nor the names of contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

add_library(libcua INTERFACE)
target_link_libraries(libcua INTERFACE ${CUDA_LIBRARIES})
target_include_directories(libcua INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cudaArray>
  $<INSTALL_INTERFACE:include>
)

# Installation code following largely from:
# https://stackoverflow.com/questions/47718485/

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/libcuaConfigVersion.cmake"
  VERSION ${LIBCUA_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(TARGETS libcua
  EXPORT libcuaTargets
  LIBRARY DESTINATION lib COMPONENT Runtime
  ARCHIVE DESTINATION lib COMPONENT Development
  RUNTIME DESTINATION bin COMPONENT Runtime
  PUBLIC_HEADER DESTINATION include/cudaArray COMPONENT Development
  BUNDLE DESTINATION bin COMPONENT Runtime
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cudaArray/libcuaConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/libcuaConfig.cmake"
  INSTALL_DESTINATION lib/cmake/libcua
)

install(EXPORT libcuaTargets DESTINATION lib/cmake/libcua)
install(FILES
  "${PROJECT_BINARY_DIR}/libcuaConfigVersion.cmake"
  "${PROJECT_BINARY_DIR}/libcuaConfig.cmake"
  DESTINATION lib/cmake/libcua)
install(
  DIRECTORY ${PROJECT_SOURCE_DIR}/cudaArray/
  DESTINATION include/cudaArray
  FILES_MATCHING PATTERN "*.h")

add_custom_target(uninstall
  COMMAND ${CMAKE_COMMAND} -E remove_directory
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/libcua
  COMMAND ${CMAKE_COMMAND} -E remove_directory
    ${CMAKE_INSTALL_PREFIX}/include/cudaArray)
