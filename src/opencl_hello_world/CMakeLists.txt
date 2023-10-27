find_package(OpenCL REQUIRED)

add_library(hello-world hello-world.cpp)

target_include_directories(hello-world
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

target_link_libraries(hello-world opencl_base OpenCL::OpenCL compiler_flags)

# Add OpenCL kernel file to the target
set_source_files_properties(hello-world.cl PROPERTIES HEADER_ONLY TRUE)
add_custom_target(kernel_header DEPENDS hello-world.cl 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/hello-world.cl
    ${CMAKE_CURRENT_BINARY_DIR}/hello-world.cl)
add_dependencies(hello-world kernel_header)

add_executable(hello-world-example example.cpp)
target_link_libraries(hello-world-example hello-world)
