# Tools that link the runtime.
foreach(tool grx-smi grx-conform)
  file(GLOB ${tool}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${tool}/*.cpp")
  if(${tool}_SRC)
    add_executable(${tool} ${${tool}_SRC})
    target_link_libraries(${tool} PRIVATE grxrt)
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endif()
endforeach()

# Tools that must stay linkable without a device stack present.
#
#   grxify       a source translator: reads the API table, never calls the runtime
#   grx-sanitize runs a program and reads its report; it opens no device itself,
#                and linking one in would make a debugging aid depend on the
#                thing being debugged
#   grx-prof     same shape: it runs a program, reads GRXPROF lines, and writes
#                a trace. The device work is all in the child process.
#   grxcc        the single-source driver. It rewrites text and shells out to
#                ci/build_kernel.sh and the host compiler; it never opens a
#                device. Linking the runtime into it would mean a machine that
#                only COMPILES for GRX-G100 needs a driver installed to do it.
foreach(tool grxify grx-sanitize grx-prof grxcc)
  file(GLOB ${tool}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${tool}/*.cpp")
  if(${tool}_SRC)
    add_executable(${tool} ${${tool}_SRC})
    # Headers only -- grxcc reads grx_abi.h to build the fat-binary blob it
    # embeds. That is a compile-time dependency on the ABI, not a link-time one
    # on the runtime.
    target_include_directories(${tool} PRIVATE
                               "${CMAKE_CURRENT_SOURCE_DIR}/../include")
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endif()
endforeach()
