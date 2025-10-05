cmake_minimum_required (VERSION 3.9)
project (gipuma)

#gSLIC
IF(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)


# Support IDEs: https://cliutils.gitlab.io/modern-cmake/chapters/features/ides.html
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake-default-targets")

find_package(CUDA 6.0 REQUIRED ) # For Cuda Managed Memory and c++11
find_package(OpenCV REQUIRED )
if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(ORUtils)# 子文件夹项目进行cmake编译
include_directories(.)

# from https://en.wikipedia.org/wiki/CUDA#GPUs_supported
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_52,code=sm_52 -lineinfo -cudart=static -Xptxas -v")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -Ofast -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# gSLIC
set(GSLICR_LIB
gSLICr_Lib/engines/gSLICr_core_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
gSLICr_Lib/engines/gSLICr_core_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
gSLICr_Lib/objects/gSLICr_settings.h
gSLICr_Lib/objects/gSLICr_spixel_info.h
gSLICr_Lib/gSLICr_defines.h
gSLICr_Lib/gSLICr.h
)
list(APPEND "-std=c++11 -ftree-vectorize")
SOURCE_GROUP(engines FILES ${GSLICR_LIB})

cuda_add_library(gSLICr_lib
			${GSLICR_LIB}
			NVTimer.h
			OPTIONS -gencode arch=compute_52,code=compute_52)
target_link_libraries(gSLICr_lib ${CUDA_LIBRARY})
target_link_libraries(gSLICr_lib ${OpenCV_LIBS})


# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    gipuma
    cameraGeometryUtils.h
    vector_operations.h
    camera.h
    globalstate.h
    algorithmparameters.h
    cameraparameters.h
    linestate.h
    groundTruthUtils.h
    displayUtils.h
    fileIoUtils.h
    main.h
    gipuma.cu
    main.cpp
    )


# For linking ...
# Specify target & libraries to link it with
target_link_libraries(gipuma gSLICr_lib ${OpenCV_LIBS})
target_link_libraries(gipuma ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

include(FeatureSummary)
feature_summary(WHAT ALL)

