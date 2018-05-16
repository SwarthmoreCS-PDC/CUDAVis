// a simple example of how to use gpuDisplayData library to
// animate a cuda computation
//
// Most of the openGL-Cuda interoperability stuff is hidden
// in the GPUDisplayData class.  However, you need to think
// a little bit about writing an event driven program since
// you register animation and clean-up events with this libary
// and then run your animation.
//
// This example creates a complex number struct and uses it
// to generate and optionally animate Julia Set fractals.
//
// (danner, 2018)

#include "gpuDisplayData.h"
#include "handle_cuda_error.h"
#include "timerGPU.h"
#include <math.h>
#include <stdio.h>
#include <unistd.h>

// try changing this to different powers of 2
#define DIM 1024
#define PI 3.1415926535897932f

static void animate_julia(uchar3 *disp, void *mycudadata);
static void clean_up(void *mycudadata);
__host__ __device__ int julia(int x, int y, float re, float im);
__global__ void julia_kernel(uchar3 *data, int size, float re, float im);

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
  float re;  // real compenent of complex seed
  float im;  // imaginary compenent of complex seed
  int ticks; // counter
} my_cuda_data;

int main(int argc, char *argv[]) {

  // single var holds all program data.  This will be passed to the
  // GPUDisplayData constructor
  my_cuda_data info;
  info.size = DIM;
  info.re = -0.800;
  info.im = 0.156;
  info.ticks = 0;

  // The call to the constructor has to come before any calls to
  // cudaMalloc or other Cuda routines
  // This is part of the reason why we are passing the address of
  // a struct with fields which are ptrs to cudaMalloc'ed space
  // The other reason is that adding a level of indirection
  // is the answer to every problem.
  GPUDisplayData my_display(info.size, info.size, &info, "Fractal CUDA");

  // register a clean-up function on exit that will free
  // any dynamically allocated memory on the GPU or the CPU
  my_display.RegisterExitFunction(clean_up);

  // have the library run our CUDA animation
  my_display.AnimateComputation(animate_julia);
  return 0;
}

// cleanup function passed to AnimateComputation method.
// it is called when the program exits and should clean up
// all dynamically allocated memory in the my_cuda_data struct.
// Your clean-up function's prototype must match this
static void clean_up(void *mycudadata) { /* do nothing */ }

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
static void animate_julia(uchar3 *devPtr, void *my_data) {

  my_cuda_data *data = (my_cuda_data *)my_data;
  dim3 blocks(data->size, data->size);

  /* This example just needs a time tick to generate an image
     directly in the GPU pixel buffer on the fly. No additional
     GPU pointers are needed from the my_cuda_data struct.
     devPtr is created and passed by the GPUDisplayData class */
  float im = data->im + 0.2 * sin(data->ticks / 20.);
  float re = data->re + 0.3 * cos(data->ticks / 17.);
  GPUTimer timer;
  timer.start();
  julia_kernel<<<blocks, 1>>>(devPtr, data->size, re, im);
  printf("Frame generation time: %7.2f ms\r", timer.elapsed());
  data->ticks += 1;
}

__global__ void julia_kernel(uchar3 *optr, int size, float re, float im) {
  // map from threadIdx/BlockIdx to pixel position
  int x = blockIdx.x;
  int y = blockIdx.y;
  int offset = x + y * size;

  // now calculate the value at that position
  int juliaValue = julia(x, y, re, im);
  if (juliaValue == -1) { /* in the set */
    optr[offset].x = 99;
    optr[offset].y = 25;
    optr[offset].z = 25;
  } else if (juliaValue < 10) {
    optr[offset].x = 0;
    optr[offset].y = 255 * juliaValue / 10.;
    optr[offset].z = 0;
  } else {
    float t = juliaValue / 200.;
    optr[offset].x = 0;
    optr[offset].y = 0;
    optr[offset].z = 255 * t;
  }
}

struct cuComplex {
  float r;
  float i;
  __host__ __device__ cuComplex(float a, float b) : r(a), i(b) {}
  __host__ __device__ float magnitude2(void) { return r * r + i * i; }
  __host__ __device__ cuComplex operator*(const cuComplex &a) {
    return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
  }
  __host__ __device__ cuComplex operator+(const cuComplex &a) {
    return cuComplex(r + a.r, i + a.i);
  }
};

__host__ __device__ int julia(int x, int y, float re, float im) {
  const float scale = 1.5;
  float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
  float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

  cuComplex c(re, im);
  cuComplex a(jx, jy);

  int i = 0;
  for (i = 0; i < 200; i++) {
    a = a * a + c;
    if (a.magnitude2() > 1000)
      return i;
  }

  return -1;
}
