add_subdirectory(Stopwatches)

# Add LodePNG without editing the git submodule
add_library(lodepng STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/lodepng/lodepng.cpp
  )

target_include_directories(lodepng 
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/lodepng
  )

add_subdirectory(lodepng_wrapper)
add_subdirectory(utilities)
add_subdirectory(opencl_base)
add_subdirectory(opencl_hello_world)
add_subdirectory(occlusion_fill)
add_subdirectory(half)

add_executable(phase1 phase1.cpp)
target_link_libraries(phase1
  PRIVATE
  lodepng_wrapper
  utilities
  stopwatches
  occlusion_fill
  opencl_base
  compiler_flags
  )

add_executable(phase2 phase2.cpp)
target_link_libraries(phase2
  PRIVATE
  stopwatches
  opencl_base
  compiler_flags
  )

add_executable(phase3 phase3.cpp)
target_link_libraries(phase3
  PRIVATE
  lodepng_wrapper
  half
  utilities
  stopwatches
  occlusion_fill
  opencl_base
  compiler_flags
  )
  
add_executable(phase3_conv_f16 phase3_conv_f16.cpp)
target_link_libraries(phase3_conv_f16
  PRIVATE
  lodepng_wrapper
  half
  utilities
  stopwatches
  occlusion_fill
  opencl_base
  compiler_flags
  )

add_executable(phase4 phase4.cpp)
target_link_libraries(phase4
  PRIVATE
  lodepng_wrapper
  utilities
  stopwatches
  occlusion_fill
  opencl_base
  compiler_flags
  )

add_executable(phase5 phase5.cpp)
target_link_libraries(phase5
  PRIVATE
  lodepng_wrapper
  utilities
  stopwatches
  occlusion_fill
  opencl_base
  compiler_flags
  )

add_executable(vgg vgg.cpp)
target_link_libraries(vgg
        PRIVATE
        lodepng_wrapper
        utilities
        stopwatches
        occlusion_fill
        opencl_base
        compiler_flags
)

add_executable(imageConvert image_convert.cpp)
target_link_libraries(imageConvert
        PRIVATE
        lodepng_wrapper
        utilities
        stopwatches
        occlusion_fill
        opencl_base
        compiler_flags
)

add_executable(phase6_main phase6_main.cpp
        phase6_model.cpp
        phase6_ref.cpp)
target_link_libraries(phase6_main
        PRIVATE
        lodepng_wrapper
        utilities
        stopwatches
        occlusion_fill
        opencl_base
        compiler_flags
)

add_executable(phase6_ref phase6_ref.cpp)
target_link_libraries(phase6_ref
        PRIVATE
        lodepng_wrapper
        utilities
        stopwatches
        occlusion_fill
        opencl_base
        compiler_flags
)

# Copy all opencl kernels to the build directory
add_custom_target(copy_kernels ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  ${CMAKE_CURRENT_BINARY_DIR}/kernels
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/kernels
  )
add_dependencies(phase1 copy_kernels)
add_dependencies(phase2 copy_kernels)
add_dependencies(phase3 copy_kernels)
add_dependencies(phase3_conv_f16 copy_kernels)
add_dependencies(phase4 copy_kernels)
add_dependencies(phase5 copy_kernels)
add_dependencies(phase6_main copy_kernels)
add_dependencies(vgg copy_kernels)
add_dependencies(imageConvert copy_kernels)

