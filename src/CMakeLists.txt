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

if(GRXRT_SOURCES)
  target_sources(grxrt PRIVATE ${GRXRT_SOURCES})
else()
  # Phase 0: no implementation yet. Keep the target linkable so the build
  # graph, install rules, and pkg-config file can be validated end to end.
  target_sources(grxrt PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/runtime/placeholder.cpp")
endif()
