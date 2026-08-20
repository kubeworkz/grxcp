# Tools that link the runtime.
foreach(tool grx-smi grx-prof grx-conform)
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
foreach(tool grxify grx-sanitize)
  file(GLOB ${tool}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${tool}/*.cpp")
  if(${tool}_SRC)
    add_executable(${tool} ${${tool}_SRC})
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endif()
endforeach()
