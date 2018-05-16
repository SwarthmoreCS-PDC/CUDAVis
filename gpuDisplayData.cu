/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
// A class for openGL animation of CUDA applications
//
//
#include "gpuDisplayData.h"
#include <GL/freeglut.h>
#include <string.h>

GPUDisplayData *GPUDisplayData::gpu_disp = 0;

// the constructor takes dimensions of the openGL graphics display
// object to create, and a pointer to a struct containing ptrs
// to application-specific CUDA data that the display function
// needs in order to change bitmap values in the openGL object
GPUDisplayData::GPUDisplayData(int w, int h, void *data,
                               const char *winname = "Animation")
    : resource(NULL), width(w), height(h), quad(w, h), paused(false),
      gpu_data(data), animate_function(NULL), exit_function(NULL) {
  // init glut
  int argc = 0; // bogus args for glutInit
  char *argv = NULL;
  glutInit(&argc, &argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
  glutInitWindowSize(width, height);
  glutCreateWindow(winname);
  // Note: glutSetOption is only available with freeglut
  // Returns control from glutMainLoop to caller when user
  // closes window
  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);

  // init glew
  GLenum err = glewInit();
  if (GLEW_OK != err) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
  }

  if (!quad.init("tquad_v.glsl", "tquad_f.glsl")) {
    fprintf(stderr, "Quad init oops\n");
  }

  // Create a resource handle that allows CUDA to
  // modify OpenGL PBO created by quad
  HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, quad.getPBO(),
                                            cudaGraphicsMapFlagsNone));

  // static weirdness
  gpu_disp = this;
}

GPUDisplayData::~GPUDisplayData() {
  if (resource) {
    printf("resource should have been released by now\n");
    HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));
    resource = NULL;
  }
  if (exit_function) {
    exit_function(gpu_data);
  }
}

void GPUDisplayData::AnimateComputation(void (*anim_func)(uchar3 *, void *),
                                        void (*exit_func)(void *)) {

  animate_function = anim_func;
  ;
  if (exit_func && exit_function == NULL) {
    exit_function = exit_func;
  }
  // add callbacks on openGL events
  glutIdleFunc(animate);
  glutDisplayFunc(animate);
  glutKeyboardFunc(keyboard);
  glutCloseFunc(close);

  // call glut mainloop
  glutMainLoop();
}

// generic animate function registered with glutDisplayFunc
// it makes call to the application-specific animate function
void GPUDisplayData::animate(void) {
  uchar3 *devPtr;
  size_t size;
  GPUDisplayData *obj = GPUDisplayData::gpu_disp;

  if (!obj->resource) {
    return;
  }

  HANDLE_ERROR(cudaGraphicsMapResources(1, &obj->resource, NULL));
  HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void **)&devPtr, &size,
                                                    obj->resource));

  if (obj->animate_function) {
    obj->animate_function(devPtr, obj->gpu_data);
  }
  HANDLE_ERROR(cudaGraphicsUnmapResources(1, &obj->resource, NULL));

  glClearColor(1.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  obj->quad.draw();
  // glDrawPixels( obj->width, obj->height, GL_RGB, GL_UNSIGNED_BYTE, 0 );
  glutSwapBuffers();
}

void GPUDisplayData::keyboard(unsigned char key, int x, int y) {
  GPUDisplayData *obj = GPUDisplayData::gpu_disp;
  /* allow safe quit with q */
  if (key == 'q') {
    glutLeaveMainLoop();
  }
  if (key == ' ') {
    obj->paused = !obj->paused;
    if (obj->paused) {
      glutIdleFunc(NULL);
    } else {
      glutIdleFunc(animate);
    }
  }
}

void GPUDisplayData::close() {
  GPUDisplayData *obj = GPUDisplayData::gpu_disp;
  if (obj->resource) {
    HANDLE_ERROR(cudaGraphicsUnregisterResource(obj->resource));
    obj->resource = NULL;
  }
}
