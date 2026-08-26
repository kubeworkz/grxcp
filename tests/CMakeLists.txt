enable_testing()

# Unit tests link libgrxrt. With GRXCP_USE_MOCK_DRIVER=ON that library already
# carries the mock driver, so these run anywhere; with it OFF they run against
# whatever backend $VORTEX_DRIVER selects.
file(GLOB UNIT_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
foreach(src ${UNIT_SRC})
  get_filename_component(name ${src} NAME_WE)

  # Some unit tests call the mock driver's inspection helpers (grxmock_*) to
  # assert on what the runtime SUBMITTED, not just on what it returned -- a
  # launch's grid dimensions, the argument blob's size. Those symbols exist only
  # in tests/mock, so with a real driver these do not link. Skipped by reading
  # the source rather than by a hand-kept list, because a hand-kept list is one
  # more thing to forget when a test starts using the mock.
  file(READ ${src} _src_text)
  if(_src_text MATCHES "grxmock_" AND NOT GRXCP_USE_MOCK_DRIVER)
    message(STATUS "grxcp: skipping ${name} -- needs -DGRXCP_USE_MOCK_DRIVER=ON")
    continue()
  endif()

  # A unit test may also ask a LIBRARY a question that needs no device kernel --
  # test_dispatch.cpp asks grxBLAS which engine a GEMM would land on, which is
  # pure host routing. Linked by what it includes, the same rule the library
  # gates below use, and for the same reason: a filename convention is one
  # rename away from silently not building a test.
  set(_unit_libs "")
  foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
    if(TARGET ${lib} AND _src_text MATCHES "include[ \t]*<grx/${lib}\\.h>")
      list(APPEND _unit_libs ${lib})
    endif()
  endforeach()

  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE ${_unit_libs} grxrt)
  # A few unit tests include the driver's own VX_types.h to check GRXCP's view
  # of a descriptor field for field against the definition it has to match. That
  # header lives in the sysroot's KERNEL include directory, not the runtime one,
  # and it is deliberately not on grxrt's public interface: a program using
  # GRXCP should never need it.
  if(VORTEX_KERNEL_FOUND)
    target_include_directories(${name} PRIVATE ${VORTEX_KERNEL_INCLUDE_DIRS})
  endif()
  add_test(NAME ${name} COMMAND ${name})
  # Exit 77 is the skip convention: no device present is not a failure.
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Library gates: tests/libs/test_<lib>.cpp is the numerical gate for library
# <lib>, and is built only if that library was. They need a compiled device
# kernel, which this build does not produce (see src/libs/CMakeLists.txt); the
# gate skips with 77 when the kernel is missing rather than reporting a
# numerical failure against a file nobody compiled.
#
# Which libraries a gate links is read from what it INCLUDES, not from its file
# name. The name is a bad key: test_grxblas_ex.cpp and test_grxblas_l12.cpp both
# test grxblas, and stripping "test_" gives two library names that do not exist,
# so both gates silently vanished from the cmake build. A gate that exercises two
# libraries at once -- test_libs_together.cpp does, deliberately -- has no single
# name to strip down to at all.
file(GLOB LIB_TEST_SRC CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libs/*.cpp")
foreach(src ${LIB_TEST_SRC})
  get_filename_component(name ${src} NAME_WE)
  file(READ ${src} _src_text)

  set(_libs "")
  foreach(lib grxblas grxdnn grxfft grxrand grxsparse)
    if(TARGET ${lib} AND _src_text MATCHES "include[ \t]*<grx/${lib}\\.h>")
      list(APPEND _libs ${lib})
    endif()
  endforeach()

  if(_libs)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/unit)
    target_link_libraries(${name} PRIVATE ${_libs} grxrt)
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
