option(BUILD_UTILITIES "Build the utilities library" ON)

if (NOT BUILD_UTILITIES)
    return()
endif()

find_package(OpenCL REQUIRED)
find_package(OpenMP REQUIRED)

add_library(utilities utils.cpp)

target_include_directories(utilities
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

# link our compiler flags interface library
target_link_libraries(utilities OpenCL::OpenCL OpenMP::OpenMP_CXX compiler_flags)

add_executable(utilities_example example.cpp)
target_link_libraries(utilities_example utilities compiler_flags)
