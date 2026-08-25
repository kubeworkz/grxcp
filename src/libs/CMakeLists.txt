# grxBLAS / grxDNN / grxFFT / grxRAND / grxSPARSE.
#
# Library kernels are compiled with the native device toolchain (VOLT), not
# through SPIR-V, because they use tensor-core and async-copy intrinsics that
# SPIR-V cannot reach. They ship as prebuilt .vxbin modules loaded through
# grxModuleLoadData -- the same way vendor BLAS libraries ship precompiled
# hand-tuned kernels behind a host API. See roadmap phase 3.
#
# The kernels themselves are NOT built here: they need a cross toolchain this
# project does not require its host build to have. Build them with
# ci/build_kernel.sh; the libraries locate them at runtime (see grxblas.h).
foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
  file(GLOB ${lib}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${lib}/*.cpp")
  if(${lib}_SRC)
    add_library(${lib} SHARED ${${lib}_SRC})
    target_link_libraries(${lib} PUBLIC grxrt)
    install(TARGETS ${lib} LIBRARY DESTINATION lib)
  endif()
endforeach()

# grxDNN calls grxBLAS, and only grxDNN does.
#
# Attention is two batched GEMMs with a mask and a softmax between them, so
# grxdnnAttentionForward is a grxBLAS caller (grxdnn.h says so where the op is
# declared). PUBLIC rather than PRIVATE because grxdnn.h includes grxblas.h --
# a program that includes one gets the other's declarations, and its link line
# should match.
#
# This is a real dependency between two libraries that are otherwise siblings,
# so it is stated once here rather than folded into the loop above, where it
# would read as though every library had it.
if(TARGET grxdnn AND TARGET grxblas)
  target_link_libraries(grxdnn PUBLIC grxblas)
endif()
