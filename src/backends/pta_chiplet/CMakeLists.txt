# pta_chiplet -- the PTA chiplet's digital twin (board plan, step X5).
#
# The chiplet is reached through the GPU's BAR over CXL.io
# (docs/designs/pta_chiplet_regmap.md), and there is no chiplet. This directory
# holds a C model that presents that map with grx930's error model behind it,
# and the gate that holds it to the map and to the model.
#
# Nothing here goes into libgrxrt. The twin is a model for bring-up and for
# tests, like tests/mock, and a runtime that linked it could report it as a
# device. What reports it, and how, is the board plan's S4.
#
# It needs no NPU flag: the gate borrows the c930 backend's register reader to
# check that one driver addresses both maps, and compiles that file itself.
add_library(pta_chiplet_twin STATIC
    pta_chiplet_twin.c
    ${CMAKE_SOURCE_DIR}/third_party/grx930/pta_tile_model.c
)
target_include_directories(pta_chiplet_twin PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/grx930
)

add_executable(test_pta_chiplet_twin
    test_pta_chiplet_twin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/../npu_c930/npu_c930.cpp
)
target_include_directories(test_pta_chiplet_twin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../npu_c930
)
target_link_libraries(test_pta_chiplet_twin PRIVATE pta_chiplet_twin)

add_test(NAME pta_chiplet_twin COMMAND test_pta_chiplet_twin)
