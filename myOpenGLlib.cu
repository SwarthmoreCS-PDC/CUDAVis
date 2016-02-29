// A class for openGL animation of CUDA applications 
// 
//
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "cuda.h"
#include "cuda_gl_interop.h"


#include "myOpenGLlib.h"
#include <string.h>


GPUDisplayData  *GPUDisplayData::gpu_disp = 0;


// the constructor takes dimensions of the openGL graphics display 
// object to create, and a pointer to a struct containing ptrs 
// to application-specific CUDA data that the display function
// needs in order to change bitmap values in the openGL object
GPUDisplayData::GPUDisplayData(int w, int h, void *data, 
    const char *winname ="Animation") :
  bufferObj(0), resource(NULL), width(w), height(h), quad(),
  gpu_data(data), animate_function(NULL), exit_function(NULL)
{
  // init glut
  int argc = 0;   // bogus args for glutInit 
  char *argv = NULL;
  glutInit(&argc, &argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA ); 
  glutInitWindowSize( width, height ); 
  glutCreateWindow( winname );
  // Note: glutSetOption is only available with freeglut
  // Returns control from glutMainLoop to caller when user
  // closes window
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
              GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  //init glew
  GLenum err = glewInit();
  if (GLEW_OK != err){
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }


  // create an OpenGL buffer for pixel texture data
  glGenBuffers( 1, &bufferObj );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, bufferObj );
  glBufferData( GL_PIXEL_UNPACK_BUFFER, width * height * 3, 
      NULL, GL_DYNAMIC_DRAW );

  // Create a resource handle that allows CUDA to 
  // modify OpenGL bufferObj created above
  HANDLE_ERROR( cudaGraphicsGLRegisterBuffer( &resource, bufferObj, 
        cudaGraphicsMapFlagsNone ) );

  quad.init();

  // static weirdness
  gpu_disp = this;
}


GPUDisplayData::~GPUDisplayData() {

  printf("in GPUDisplayData::destructor\n");
  if (resource){
    HANDLE_ERROR( cudaGraphicsUnregisterResource(resource) );
    resource = NULL;
  }
  if (bufferObj != 0) {
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
    glDeleteBuffers( 1, &bufferObj );
    bufferObj = 0;
  }
}


void GPUDisplayData::AnimateComputation( 
    void (*anim_func)(uchar3 *, void *), void (*exit_func)(void *)) 
{

  animate_function = anim_func;;
  if(exit_func && exit_function==NULL){
    exit_function = exit_func;
  }
  // add callbacks on openGL events
  glutIdleFunc(animate);
  glutDisplayFunc(animate);

  /* not needed, destructor called when glutSetOption
     tells glut to return from main loop on window close */
  //atexit(clean_up);   // register function to clean up state on exit

  // call glut mainloop
  glutMainLoop();
}

// cleanup function for call to atexit
void GPUDisplayData::clean_up(void) {

  printf("in GPUDisplayData::clean_up\n");
  /* handled by destructor now */
}

// generic animate function registered with glutDisplayFunc
// it makes call to the application-specific animate function
void GPUDisplayData::animate(void) {
  uchar3 *devPtr;
  size_t size;
  GPUDisplayData *obj = GPUDisplayData::get_gpu_obj();


  HANDLE_ERROR( cudaGraphicsMapResources( 1, &obj->resource, NULL ) ) ;
  HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer( (void**)&devPtr, 
        &size, obj->resource) );

  if(obj->animate_function) { 
    obj->animate_function(devPtr, obj->gpu_data);
  }
  HANDLE_ERROR( cudaGraphicsUnmapResources( 1, &obj->resource, NULL ) );

  glClearColor( 1.0, 0.0, 0.0, 1.0 );
  glClear( GL_COLOR_BUFFER_BIT );
  obj->quad.draw();
  //glDrawPixels( obj->width, obj->height, GL_RGB, GL_UNSIGNED_BYTE, 0 );
  glutSwapBuffers();

}
