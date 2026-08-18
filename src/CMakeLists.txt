# Runtime sources. The library target is declared at the top level so tools and
# tests can link it before this directory is processed.
file(GLOB GRXRT_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/runtime/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/backends/gpu_g100/*.cpp")

if(GRXCP_ENABLE_NPU)
  file(GLOB GRXRT_NPU_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/backends/npu_c930/*.cpp")
  list(APPEND GRXRT_SOURCES ${GRXRT_NPU_SOURCES})
endif()

if(NOT GRXRT_SOURCES)
  message(FATAL_ERROR "grxcp: no runtime sources found under src/runtime/")
endif()
target_sources(grxrt PRIVATE ${GRXRT_SOURCES})
