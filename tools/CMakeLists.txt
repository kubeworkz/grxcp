# Tools that link the runtime.
foreach(tool grx-smi grx-prof grx-sanitize grx-conform)
  file(GLOB ${tool}_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${tool}/*.cpp")
  if(${tool}_SRC)
    add_executable(${tool} ${${tool}_SRC})
    target_link_libraries(${tool} PRIVATE grxrt)
    install(TARGETS ${tool} RUNTIME DESTINATION bin)
  endif()
endforeach()

# grxify is a source translator: it reads the API table and never calls the
# runtime, so it must stay linkable without a device stack present.
file(GLOB grxify_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/grxify/*.cpp")
if(grxify_SRC)
  add_executable(grxify ${grxify_SRC})
  install(TARGETS grxify RUNTIME DESTINATION bin)
endif()
