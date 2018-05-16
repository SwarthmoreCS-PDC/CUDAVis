/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
// a simple example of how to use gpuDisplayData library to
// animate a cuda computation
//
// Most of the openGL-Cuda interoperability stuff is hidden
// in the GPUDisplayData class.  However, you need to think
// a little bit about writing an event driven program since
// you register animation and clean-up events with this libary
// and then run your animation.
//
// This example shows how to use our class to recreate the ripple animation
// from CUDA by Example (Section 5.2.2)
// (danner, 2018)

#include <unistd.h>
#include <stdio.h>
#include "gpuDisplayData.h"
#include "handle_cuda_error.h"

// try changing this to different powers of 2
#define DIM 1024
#define PI 3.1415926535897932f

static void animate_ripple(uchar3 *disp, void *mycudadata);
static void clean_up(void* mycudadata);
__global__ void  ripple(uchar3 *data, int size, int ticks);

/* The GPUDisplayData class will automatically create
   an RGB buffer of a given width and height for you on
   the GPU. Writing to this buffer in a CUDA kernel will
   modify the image in the animation loop.

   If your program needs additional GPU data, or dynamically
   allocated CPU data, use the following struct to
   store the needed information.
*/
typedef struct my_cuda_data {
  int size;  // width and height of image
  int ticks; // a time tick for updating the animation
} my_cuda_data;


int main(int argc, char *argv[])  {

  // single var holds all program data.  This will be passed to the
  // GPUDisplayData constructor
  my_cuda_data info;
  info.size=DIM;
  info.ticks=0;

  // The call to the constructor has to come before any calls to
  // cudaMalloc or other Cuda routines
  // This is part of the reason why we are passing the address of
  // a struct with fields which are ptrs to cudaMalloc'ed space
  // The other reason is that adding a level of indirection
  // is the answer to every problem.
  GPUDisplayData my_display(info.size, info.size, &info, "Ripple CUDA");

  // register a clean-up function on exit that will free
  // any dynamically allocated memory on the GPU or the CPU
  my_display.RegisterExitFunction(clean_up);

  // have the library run our CUDA animation
  my_display.AnimateComputation(animate_ripple);
  return 0;
}

// cleanup function passed to AnimateComputation method.
// it is called when the program exits and should clean up
// all dynamically allocated memory in the my_cuda_data struct.
// Your clean-up function's prototype must match this
static void clean_up(void* mycudadata) {
  /* do nothing */
}

// amimate function passed to AnimateComputation:
// this function will be called by openGL's dislplay function.
// It can contain code that runs on the CPU and also calls to
// to CUDA kernel code to do a computation and to change the
// display the results using openGL...you need to change the
// display color values based on the application values
//
// devPtr: is pointer into openGL buffer of rgb values (but
//         the field names are x,y,z)
// my_data: is pointer to our cuda data that we passed into the
//          constructor
//
// your animate function prototype must match this one:
static void animate_ripple(uchar3 *devPtr, void *my_data) {

  int tdim = 8;
  my_cuda_data *data = (my_cuda_data *)my_data;
  dim3 blocks(data->size/tdim, data->size/tdim);
  dim3 threads_block(tdim, tdim);

  /* This example just needs a time tick to generate an image
     directly in the GPU pixel buffer on the fly. No additional
     GPU pointers are needed from the my_cuda_data struct.
     devPtr is created and passed by the GPUDisplayData class */
  ripple<<<blocks,threads_block>>>(devPtr, data->size, data->ticks);
  data->ticks += 2;
}

__global__ void ripple( uchar3* optr, int size, int ticks ) {
    // map from threadIdx/BlockIdx to pixel position
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * size;

    // compute distance from center of image
    float fx = x - size/2;
    float fy = y - size/2;
    float d = sqrtf( fx * fx + fy * fy );
    unsigned char grey = (unsigned char)(128.0f + 127.0f *
                                         cos(d/10.0f - ticks/7.0f) /
                                         (d/10.0f + 1.0f));
    optr[offset].x = grey;
    optr[offset].y = grey;
    optr[offset].z = grey;

}
