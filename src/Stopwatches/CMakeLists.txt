option(BUILD_STOPWATCHES "Build the stopwatches library" ON)

if (NOT BUILD_STOPWATCHES)
    return()
endif()

add_library(stopwatches Stopwatches.cpp)

target_include_directories(stopwatches
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    )

# link our compiler flags interface library
target_link_libraries(stopwatches compiler_flags)

add_executable(stopwatches_example example.cpp)
target_link_libraries(stopwatches_example stopwatches compiler_flags)
