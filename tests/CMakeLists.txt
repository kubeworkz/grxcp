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

# The device-property contract must hold on an FPGA backend too, where managed
# memory has to be gated off. Same binary, different backend selection.
if(TARGET test_device_props)
  add_test(NAME test_device_props_fpga COMMAND test_device_props)
  set_tests_properties(test_device_props_fpga PROPERTIES
    ENVIRONMENT "VORTEX_DRIVER=xrt" SKIP_RETURN_CODE 77)
endif()
