cmake_minimum_required(VERSION 3.10)
project(kisa C)

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Check processor type and OS
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(STATUS "Building for ARM64 with NEON support")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DHAVE_NEON -D__aarch64__")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # Check for AVX-512 support on x86_64
    include(CheckCCompilerFlag)
    check_c_compiler_flag("-mavx512f -mavx512dq -mavx512bw -mavx512vl" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        message(STATUS "Building for x86_64 with AVX-512 support")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512dq -mavx512bw -mavx512vl -DHAVE_AVX512")
    else()
        message(STATUS "Building for x86_64 without AVX-512 support")
    endif()
else()
    message(WARNING "Unknown processor architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Add optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

# Add source files
set(SOURCE_FILES
    src/core/vector_ops.c
    src/core/avx512_ops.c
    src/core/instruction.c
    src/core/execution.c
    src/core/control_flow.c
    src/core/decoder.c
)

# Add library
add_library(kisa STATIC ${SOURCE_FILES})

# Add include directories
target_include_directories(kisa PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link with math library
target_link_libraries(kisa PRIVATE m)

# Create test executables
add_executable(test_vector_ops tests/test_vector_ops.c)
target_include_directories(test_vector_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_vector_ops PRIVATE kisa m)

add_executable(test_control_flow tests/test_control_flow.c)
target_include_directories(test_control_flow PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(test_control_flow PRIVATE kisa m)

add_executable(kisa_test tests/test_main.c)
target_include_directories(kisa_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(kisa_test PRIVATE kisa m)

add_executable(perf_test tests/perf_test.c)
target_include_directories(perf_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(perf_test PRIVATE kisa m)

# Add application executables
add_executable(orderbook_edm_analysis src/apps/orderbook_edm_analysis.c)
target_include_directories(orderbook_edm_analysis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(orderbook_edm_analysis PRIVATE kisa m)

add_executable(llm_calc src/apps/llm_calc.c)
target_include_directories(llm_calc PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(llm_calc PRIVATE kisa m)

# Enable testing
enable_testing()
add_test(NAME vector_ops_test COMMAND test_vector_ops)
add_test(NAME control_flow_test COMMAND test_control_flow)
add_test(NAME kisa_test COMMAND kisa_test)
add_test(NAME performance_test COMMAND perf_test)

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Source files: ${SOURCE_FILES}")
message(STATUS "Include dirs: ${CMAKE_CURRENT_SOURCE_DIR}/include") 