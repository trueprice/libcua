# CudaArray: header-only library for interfacing with CUDA array-type objects
# Author: True Price <jtprice at cs.unc.edu>
#
# BSD License
# Copyright (C) 2017  The University of North Carolina at Chapel Hill
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

include_directories("${CUDA_INCLUDE_DIRS}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
  -std=c++11 -O3 --expt-extended-lambda --expt-relaxed-constexpr
  -Wno-deprecated-gpu-targets)
  
macro (LIBCUA_TEST NAME)
  cuda_add_executable(${NAME}_test ${NAME}_test.cu)
  target_include_directories(${NAME}_test PUBLIC ${CMAKE_CURRENT_LIST_DIR})
  target_link_libraries(${NAME}_test libcua gtest gtest_main)
  add_test(
    NAME ${NAME}_test COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}_test)
endmacro (LIBCUA_TEST)

libcua_test(cudaArray2D)
libcua_test(cudaArray3D)
libcua_test(cudaSurface2D)
libcua_test(cudaSurface3D)
libcua_test(cudaTexture2D)
libcua_test(random)
