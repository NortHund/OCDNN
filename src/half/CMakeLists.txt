option(BUILD_UTILITIES "Build the half library" ON)

if (NOT BUILD_UTILITIES)
    return()
endif()

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_library(half half.cpp)

target_include_directories(half
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

# link our compiler flags interface library
target_link_libraries(utilities OpenCL::OpenCL OpenMP::OpenMP_CXX compiler_flags)
