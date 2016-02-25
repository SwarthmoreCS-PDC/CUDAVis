// a simple example of how to use myopengllib library to 
// animate a cuda computation
//
// Most of the openGL-Cuda interoperability stuff is hidden
// in the GPUDisplayData class.  However, you need to think
// a little bit about writing an event driven program since
// you register animation and clean-up events with this libary
// and then run your animation.
//
// This example application doesn't do anything special, but it 
// shows how to use the GPUDisplayData library and how to write animate 
// and clean_up functions to pass to GPUDisplayData.AnimateComputation.
//
// (newhall, 2016)

#include <unistd.h>
#include <stdio.h>
#include "myopengllib.h"
#include "handle_cuda_error.h"

// try changing this to different powers of 2
#define DIM 512

static void animate_simple(uchar3 *disp, void *mycudadata);
static void clean_up(void* mycudadata);
__global__ void int_to_color( uchar3 *optr, 
    const int *my_cuda_data, int ncols);
__global__ void  simplekernel(int *data, int ncols); 



// if your program needs more GPU data, use a struct
// with fields for each value needed.
typedef struct my_cuda_data {
  int *dev_grid;
  int *cpu_grid;
  int rows;
  int cols;
} my_cuda_data;


int main(int argc, char *argv[])  {
 
  // single var holds all program data.  This will be passed to the
  // GPUDisplayData constructor 
  my_cuda_data info;
  int* matrix;
  int rows=DIM;
  int cols=DIM;
   
  matrix = (int*) malloc(rows*cols*sizeof(int));
  info.rows=rows;
  info.cols=cols;
  info.cpu_grid = matrix;

  //simple_prog_data.cpu_grid=smat;

  // The call to the constructor has to come before any calls to 
  // cudaMalloc or other Cuda routines
  // This is part of the reason why we are passing the address of 
  // a struct with fields which are ptrs to cudaMalloc'ed space
  // The other reason is that adding a level of indirection 
  // is the answer to every problem.
  GPUDisplayData my_display(info.cols, info.rows, &info, "Simple openGL-Cuda");

  // initialize application data 
  for(int i =0; i < info.rows; i++) {
    for(int j =0; j < info.cols; j++) {
      matrix[i*info.cols+j] = j;
    }
  }

  // allocate memory space for our application data on the GPU
  HANDLE_ERROR(cudaMalloc((void**)&info.dev_grid, 
        sizeof(int)*info.rows*info.cols) ) ;

  // copy the initial data to the GPU
  HANDLE_ERROR (cudaMemcpy(info.dev_grid, matrix, 
        sizeof(int)*info.rows*info.cols, cudaMemcpyHostToDevice) ) ;

  // register a clean-up function on exit that will call cudaFree 
  // on any cudaMalloc'ed space
  my_display.RegisterExitFunction(clean_up); 

  // have the library run our Cuda animation
  my_display.AnimateComputation(animate_simple);

  return 0;
}

// cleanup function passed to AnimateComputatin method.
// it is called when the program exits and should clean up
// all cudaMalloc'ed state.
// Your clean-up function's prototype must match this, which is 
// why simple_prog_data needs to be a global
static void clean_up(void* mycudadata) {
  my_cuda_data* info = (my_cuda_data*) mycudadata;
  HANDLE_ERROR(cudaFree(info->dev_grid) );
  free( info->cpu_grid );

}

// amimate function passed to AnimateComputation: 
// this function will be called by openGL's dislplay function.
// It can contain code that runs on the CPU and also calls to
// to CUDA kernel code to do a computation and to change the
// display the results using openGL...you need to change the
// display color values based on the application values
// 
// devPtr: is pointer into openGL buffer of rgba values (but
//         the field names are x,y,z,w
// my_data: is pointer to our cuda data that we passed into the 
//          constructor
// 
// your animate function prototype must match this one:
static void animate_simple(uchar3 *devPtr, void *my_data) {

  my_cuda_data *simple_data = (my_cuda_data *)my_data;
  dim3 blocks(simple_data->cols/8, simple_data->rows/8, 1); 
  dim3 threads_block(8, 8, 1); 

  // comment out the for loop to do a display update every 
  // execution of simplekernel
  for(int i=0; i < 90; i++) 
    simplekernel<<<blocks,threads_block>>>(
        simple_data->dev_grid, simple_data->cols); 

  int_to_color<<<blocks,threads_block>>>(devPtr, 
      simple_data->dev_grid,  simple_data->cols); 

  // I needed to slow it down:
  usleep(50000);

}

// a kernel to set the color the opengGL display object based 
// on the cuda data value
//  
//  optr: is an array of openGL RGB pixels, each is a 
//        4-tuple (x:red, y:green, z:blue, w:opacity) 
//  my_cuda_data: is cuda 2D array of ints
__global__ void int_to_color( uchar3 *optr, const int *my_cuda_data, int cols ) {

    // get this thread's block position to map into
    // location in opt and my_cuda_data
    // the x and y values depend on how you parallelize the
    // kernel (<<<blocks, threads>>>).  
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  
    int offset = x + y*cols;

    // change this pixel's color value based on some strange
    // functions of the my_cuda_data value
    optr[offset].x = (my_cuda_data[offset]+10)%255;  // R value
    optr[offset].y = (my_cuda_data[offset]+100)%255; // G value
    optr[offset].z = (my_cuda_data[offset]+200)%255; // B value
}

// a simple cuda kernel: cyclicly increases a points value by 10
//  data: a "2D" array of int values
__global__ void  simplekernel(int *data, int ncols) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  
    int offset = x + y*ncols;

    data[offset] = (data[offset] + 10)%1000;
}


