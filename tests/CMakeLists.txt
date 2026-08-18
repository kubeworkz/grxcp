enable_testing()

# Unit tests link libgrxrt. With GRXCP_USE_MOCK_DRIVER=ON that library already
# carries the mock driver, so these run anywhere; with it OFF they run against
# whatever backend $VORTEX_DRIVER selects.
file(GLOB UNIT_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
foreach(src ${UNIT_SRC})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE grxrt)
  add_test(NAME ${name} COMMAND ${name})
  # Exit 77 is the skip convention: no device present is not a failure.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Library gates: tests/libs/test_<lib>.cpp is the numerical gate for library
# <lib>, and is built only if that library was. They need a compiled device
# kernel, which this build does not produce (see src/libs/CMakeLists.txt); the
# gate skips with 77 when the kernel is missing rather than reporting a
# numerical failure against a file nobody compiled.
file(GLOB LIB_TEST_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp")
foreach(src ${LIB_TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  string(REGEX REPLACE "^test_" "" lib ${name})
  if(TARGET ${lib})
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
    target_link_libraries(${name} PRIVATE ${lib} grxrt)
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endforeach()

# The device-property contract must hold on an FPGA backend too, where managed
# memory has to be gated off. Same binary, different backend selection.
if(TARGET test_device_props)
  add_test(NAME test_device_props_fpga COMMAND test_device_props)
  set_tests_properties(test_device_props_fpga PROPERTIES
    ENVIRONMENT "VORTEX_DRIVER=xrt" SKIP_RETURN_CODE 77)
endif()
