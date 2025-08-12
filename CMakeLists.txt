cmake_minimum_required(VERSION 3.16)
project(sk0pe_demos 
    VERSION 1.0.0
    LANGUAGES C CXX
)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Cross-platform output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific settings
if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    # macOS specific flags if needed
elseif(UNIX)
    # Linux/Unix specific flags if needed
endif()

# Function to add a demo executable
function(add_demo name)
    add_executable(${name} demos/${name}.c)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/demos)
    
    # Set output name to avoid conflicts
    set_target_properties(${name} PROPERTIES OUTPUT_NAME "${name}")
    
    # Add debug symbols for better analysis
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(MSVC)
            target_compile_options(${name} PRIVATE /Zi)
            target_link_options(${name} PRIVATE /DEBUG)
        else()
            target_compile_options(${name} PRIVATE -g)
        endif()
    endif()
    
    # Disable optimizations for clearer symbolic execution
    if(MSVC)
        target_compile_options(${name} PRIVATE /Od)
    else()
        target_compile_options(${name} PRIVATE -O0)
    endif()
    
    # Platform-specific linking
    if(UNIX AND NOT APPLE)
        target_link_libraries(${name} PRIVATE m)  # Math library on Linux
    endif()
endfunction()

# Add demos
add_demo(simple_password)
add_demo(branching_paths)
add_demo(array_bounds)
add_demo(simple_math)
add_demo(multi_stage_validator)
add_demo(function_calls)

# Create a custom target to build all demos
add_custom_target(demos ALL DEPENDS 
    simple_password
    branching_paths
    array_bounds
    simple_math
    multi_stage_validator
    function_calls
)

# Print build information
message(STATUS "Building Demos")
message(STATUS "Platform: ${CMAKE_SYSTEM_NAME}")
message(STATUS "Compiler: ${CMAKE_C_COMPILER_ID}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")