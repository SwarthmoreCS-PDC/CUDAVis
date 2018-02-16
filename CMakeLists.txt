cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "--std=c++11 -g -Wall")

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules) #for FindGLEW.cmake


FIND_PACKAGE(OpenGL REQUIRED) #for graphics
FIND_PACKAGE(GLEW REQUIRED) #for graphics

#required CMake options for QT Apps
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)

#for CUDA
find_package(CUDA QUIET REQUIRED)
cuda_include_directories(${CUDA_INC} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CUDA_SDK}/lib/)

#get symlink_data macro
include(${CMAKE_SOURCE_DIR}/cmake_modules/helpers.cmake)
include_directories(./)


set(UTIL_HEADERS
  timerBase.h timerCPU.h
  texturedQuad.h
  shaderHelpers.h
  )

set(UTIL_SRC
  timerCPU.cpp
  texturedQuad.cpp
  shaderHelpers.cpp
  )

set(GPU_UTIL_HEADERS
  handle_cuda_error.h
  timerBase.h timerGPU.h
  myOpenGLlib.h
  )

set(GPU_UTIL_SRC
  handle_cuda_error.cu
  timerGPU.cu
  myOpenGLlib.cu
  )

add_library(myutil ${UTIL_HEADERS} ${UTIL_SRC})
cuda_add_library(mycudautil ${GPU_UTIL_HEADERS} ${GPU_UTIL_SRC})

set(MYLIBS mycudautil myutil)
set(GLLIBS ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GLEW_LIBRARY})
set(GPROG simple)


foreach(PROG ${GPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG}
    ${GLLIBS}  m ${MYLIBS})
endforeach()
symlink_data(shaders *.glsl)

set(RIPPHEADERS cpu_anim.h cpu_anim.h)
cuda_add_executable(ripple ripple.cu ${RIPHEADERS})
target_link_libraries(ripple
    ${GLLIBS} m ${MYLIBS})
