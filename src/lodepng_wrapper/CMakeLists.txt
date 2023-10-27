add_library(lodepng_wrapper lodepng_wrapper.cpp)

target_include_directories(lodepng_wrapper
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(lodepng_wrapper lodepng compiler_flags)

add_executable(lodepng_wrapper_example example.cpp)
target_link_libraries(lodepng_wrapper_example lodepng_wrapper compiler_flags)
