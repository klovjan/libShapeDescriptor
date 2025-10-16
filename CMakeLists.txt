set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.5)
project(naive_gpu LANGUAGES CXX CUDA)

add_definitions(-DspinImageWidthPixels=96)
add_subdirectory(lib/libShapeDescriptor libShapeDescriptor)

enable_language(CUDA)

# find_package(CUDA REQUIRED)
find_package(OpenMP)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # In case of omp.h missing, run brew install libomp
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    include_directories(/usr/local/include /opt/homebrew/include)
endif()

include_directories (
        .
        src/
        ${OPENMP_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# set(COMMON_HEADERS
#     ${PROJECT_SOURCE_DIR}/src/some_common_header.h  # optional, replace or remove
# )

# file (GLOB_RECURSE LIBRARY_HEADERS             src/*.h)
# file (GLOB_RECURSE LIBRARY_SOURCES             src/main.cpp)

# naive_gpu executable
add_executable(naive_gpu src/naive_gpu.cu ${LIBRARY_HEADERS})
target_link_libraries(naive_gpu ShapeDescriptor fast-lzma2 OpenMP::OpenMP_CXX)
set_property(TARGET naive_gpu PROPERTY CUDA_ARCHITECTURES native)

# test_gpu executable
add_executable(test_gpu src/test_gpu.cpp ${LIBRARY_HEADERS})
target_link_libraries(test_gpu ShapeDescriptor fast-lzma2 OpenMP::OpenMP_CXX)
set_property(TARGET test_gpu PROPERTY CUDA_ARCHITECTURES native)
