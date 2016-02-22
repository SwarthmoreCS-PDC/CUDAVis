cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_CXX_FLAGS "-g -Wall")

set (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules) #for FindGLEW.cmake


FIND_PACKAGE(OpenGL REQUIRED) #for graphics
FIND_PACKAGE(GLEW REQUIRED) #for graphics

#required CMake options for QT Apps
SET(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(GLUT REQUIRED)
find_package(Threads REQUIRED)

#for window widgets
FIND_PACKAGE(Qt5Widgets REQUIRED)
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5OpenGL REQUIRED)
INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS}) 
ADD_DEFINITIONS(${Qt5Widgets_DEFINITIONS})
SET(CMAKE_CXX_FLAGS 
    "${CMAKE_CXX_FLAGS} ${Qt5Widgets_EXECUTABLE_COMPILE_FLAGS}")

#for CUDA
find_package(CUDA QUIET REQUIRED) 
cuda_include_directories(${CUDA_INC} ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(${CUDA_SDK}/lib/)

#get symlink_data macro
#include(${CMAKE_SOURCE_DIR}/cmake_modules/helpers.cmake)
include_directories(./)


set(TIMER_CPU timer_base.h timer_cpu.h timer_cpu.cpp)
set(TIMER_GPU handle_cuda_error.h timer_base.h timer_gpu.h timer_gpu.cu)

add_library(cputimer ${TIMER_CPU})
cuda_add_library(gputimer ${TIMER_GPU})
cuda_add_library(hcudaerr handle_cuda_error.h handle_cuda_error.cu)

set(HDRS book.h cpu_anim.h cpu_bitmap.h 
  gl_helper.h)

cuda_add_library(mycuda 
  book.h  myopengllib.h
  book.cu  myopengllib.cu)

set(MYLIBS mycuda cputimer gputimer hcudaerr)
set(GLLIBS ${GLUT_LIBRARY} ${OPENGL_LIBRARY} ${GLEW_LIBRARY})
set(GPROG ripple simple)


foreach(PROG ${GPROG})
  cuda_add_executable(${PROG} ${PROG}.cu ${HDRS})
  target_link_libraries(${PROG} 
    ${GLLIBS} ${CMAKE_THREAD_LIBS_INIT} m ${MYLIBS})
endforeach()

