# grxBLAS / grxDNN / grxFFT / grxRAND / grxSPARSE.
#
# Library kernels are compiled with the native device toolchain (VOLT), not
# through SPIR-V, because they use tensor-core and async-copy intrinsics that
# SPIR-V cannot reach. They ship as prebuilt .vxbin modules loaded through
# grxModuleLoadData -- the same way vendor BLAS libraries ship precompiled
# hand-tuned kernels behind a host API. See roadmap phase 3.
foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
  file(GLOB ${lib}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${lib}/*.cpp")
  if(${lib}_SRC)
    add_library(${lib} SHARED ${${lib}_SRC})
    target_link_libraries(${lib} PUBLIC grxrt)
    install(TARGETS ${lib} LIBRARY DESTINATION lib)
  endif()
endforeach()
