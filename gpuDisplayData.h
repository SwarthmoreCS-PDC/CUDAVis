/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
// This is a class for doing openGL GPU animations of CUDA
// 2D grid applications
//
// It hides all/most of the openGL code, and exports to the user
// only a low-level pixel object to change color.  It has fields
// that contain application-specific CUDA data that are passed to
// user-defined functions that make cuda kernel calls to update program
// state and color openGL objects.
//
// To use this library:
// (0) create a new GPUDisplayData passing in a pointer to a struct
//     of cuda mem ptrs used by your application
// (1) allocate CUDA memory for your program and init it
// (2) call RegisterExitFunction method to register a clean up function
//     that will be executed on exit (i.e. cudaFree's stuff)
// (3) call AnimateComputation method passing in your main animate and do
//     cuda processing function  (it should contain kernel calls to
//     perform an iteration of the cuda computation and a call to
//     your application-specific kernel to color pixel values based
//     on your application)
//
//  (newhall, 2011)
//
#pragma once

#include <stdio.h>
#include <GL/glew.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "handle_cuda_error.h"
#include "texturedQuad.h"


class GPUDisplayData {

private:
  // openGL data structures
  cudaGraphicsResource *resource;
  int width; // used to create openGL bufferObj of some dimension
  int height;
  TexturedQuad quad; // A textured square containing the image pixels
  bool paused;       // Flag to indicate if animation is paused

  void *gpu_data; // application-specific CUDA data that is used by
                  // application-spcific drawing functions (this
                  // is just passed as param to user-supplied
                  // animate_function)
                  //
  static GPUDisplayData *gpu_disp; // horrible kludge due to glut

  // function pointers to user-supplied functions:
  // set an openGL bitmap value based on the program data values
  // an animation function will take a uchar3 provided by this
  // class and the gpu_data field value which is a pointer to
  // som GPU side CUDA data that is used by the animate function
  void (*animate_function)(uchar3 *color_value, void *cuda_data);
  // a function that cleans up application-specific, CUDA-side,
  // allocations
  void (*exit_function)(void *cuda_data);

  /* private static functions. These are all static because of how
     glut handles passing of display, keyboard and close functions
     in a very non C++-friendly way. */
  static void animate(void); // function passed to openGLDisplay
  static void keyboard(unsigned char key, int x, int y);
  static void close(void);

public:
  // the constructor takes dimensions of the openGL graphics display
  // object to create, and a pointer to a struct containing ptrs
  // to application-specific CUDA data that the display function
  // needs in order to change bitmap values in the openGL object
  GPUDisplayData(int w, int h, void *data, const char *win_name);
  ~GPUDisplayData();

  void RegisterExitFunction(void (*exit_func)(void *data)) {
    exit_function = exit_func;
  }

  // this is the main loop, the caller passes in the
  // animation function and an optional exit function
  void AnimateComputation(void (*anim_func)(uchar3 *, void *),
                          void (*exit_func)(void *) = NULL);
};
