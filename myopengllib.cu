// A class for openGL animation of CUDA applicatins 
// 
//
#include <GL/glew.h>
#include <GL/glut.h>
#include "cuda.h"
#include "cuda_gl_interop.h"


#include "myopengllib.h"
#include <string.h>


GPUDisplayData  *GPUDisplayData::gpu_disp = 0;


// the constructor takes dimensions of the openGL graphics display 
// object to create, and a pointer to a struct containing ptrs 
// to application-specific CUDA data that the display function
// needs in order to change bitmap values in the openGL object
GPUDisplayData::GPUDisplayData(int w, int h, void *data, 
    const char *winname ="Animation") 
{

  width = w;
  height = h;
  gpu_data = data;
  animate_function = 0;
  exit_function = 0;

  // find a CUDA device and set it to graphic interoperable
  cudaDeviceProp  prop;
  int dev;
  memset( &prop, 0, sizeof( cudaDeviceProp ) );
  prop.major = 1;
  prop.minor = 0;
  HANDLE_ERROR( cudaChooseDevice( &dev, &prop ) );
  cudaGLSetGLDevice( dev );

  // init glut
  int argc = 1;   // bogus args for glutInit 
  char *argv = NULL;
  glutInit(&argc, &argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA ); 
  glutInitWindowSize( width, height ); 
  glutCreateWindow( winname );

  //init glew
  GLenum err = glewInit();
  if (GLEW_OK != err){
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }


  // create an OpenGL buffer
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, 
      NULL, GL_DYNAMIC_DRAW_ARB );

  // register buffers for CUDA: 
  HANDLE_ERROR( cudaGraphicsGLRegisterBuffer( &resource, bufferObj, 
        cudaGraphicsMapFlagsNone ) );
  gpu_disp = this;
}


GPUDisplayData::~GPUDisplayData() {

  HANDLE_ERROR( cudaGraphicsUnregisterResource( resource ) );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );
  glDeleteBuffers( 1, &bufferObj );

}


void GPUDisplayData::AnimateComputation( void (*anim_func)(uchar4 *, void *)) {

  animate_function = anim_func;;

   // add callbacks on openGL events
   glutIdleFunc(animate);
   glutDisplayFunc(animate);
   atexit(clean_up);   // register function to clean up state on exit
   
   // call glut mainloop
   glutMainLoop();

}

void GPUDisplayData::AnimateComputation( void (*anim_func)(uchar4 *, void *), 
    void (*exit_func)(void *)) 
{
 
  animate_function = anim_func;;
  exit_function = exit_func;

   // add callbacks on openGL events
   glutIdleFunc(animate);
   glutDisplayFunc(animate);
   atexit(clean_up);   // register function to clean up state on exit
   
   // call glut mainloop
   glutMainLoop();
}

// cleanup function for call to atexit,
// 
void GPUDisplayData::clean_up(void) {

  //printf("in GPUDisplayData::clean_up\n");
  GPUDisplayData *obj = GPUDisplayData::get_gpu_obj();
  // unregister openGL buffer with cuda and free it
  HANDLE_ERROR( cudaGraphicsUnregisterResource(obj->resource) ); 
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 ); 
  glDeleteBuffers( 1, &obj->bufferObj );
  if(obj->exit_function) obj->exit_function(obj->gpu_data);
}

// generic animate function registered with glutDisplayFunc
// it makes call to the application-specific animate function
void GPUDisplayData::animate(void) {
  uchar4 *devPtr;
  size_t size;
  GPUDisplayData *obj = GPUDisplayData::get_gpu_obj();


  HANDLE_ERROR( cudaGraphicsMapResources( 1, &obj->resource, NULL ) ) ;
  HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
        &size, obj->resource) );

  if(obj->animate_function) { 
    obj->animate_function(devPtr, obj->gpu_data);
  }
  HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &obj->resource, NULL ) );

  glClearColor( 0.0, 0.0, 0.0, 1.0 );
  glClear( GL_COLOR_BUFFER_BIT );
  glDrawPixels( obj->width, obj->height, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
  glutSwapBuffers();

}
