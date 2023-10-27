add_library(occlusion_fill occlusion_fill.cpp)

target_include_directories(occlusion_fill
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(occlusion_fill compiler_flags)
