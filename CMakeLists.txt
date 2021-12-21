cmake_minimum_required(VERSION 2.8)
project(yolov5_tinytrt)

#opencv
set(OpenCV_DIR )
find_package( OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})

set(TENSORRT_LIB_PATH "" CACHE STRING "absolute path to tensorrt libraries")
set(TENSORRT_INCLUDE_PATH "" CACHE STRING "absolute path to tensorrt public header")

include_directories(${TENSORRT_INCLUDE_DIR})



find_package(CUDA REQUIRED)

# TensorRT
find_library(LIBNVINFER NAME nvinfer HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVINFER_PLUGIN NAME nvinfer_plugin HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVPARSERS NAME nvparsers HINTS ${TENSORRT_LIB_PATH} REQUIRED)
find_library(LIBNVONNXPARSER NAME nvonnxparser HINTS ${TENSORRT_LIB_PATH} REQUIRED)

add_subdirectory(tiny-tensorrt)
include_directories("tiny-tensorrt")
add_executable(yolov5_tinytrt main.cpp)

target_link_libraries( yolov5_tinytrt tinytrt)
target_link_libraries(yolov5_tinytrt ${LIBNVINFER})
target_link_libraries(yolov5_tinytrt ${LIBNVINFER_PLUGIN})
target_link_libraries(yolov5_tinytrt ${LIBNVPARSERS})
target_link_libraries(yolov5_tinytrt ${LIBNVONNXPARSER})
target_link_libraries(yolov5_tinytrt ${ZLIB_LIBRARIES})
target_link_libraries(yolov5_tinytrt ${OpenCV_LIBS})
