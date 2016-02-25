#ifndef _TEXTUREDQUAD_H
#define _TEXTUREDQUAD_H

#include <GL/gl.h>

class TexturedQuad {

  private:
    static const float sm_points[];
    static const char* sm_vertex_shader; 
    static const char* sm_fragment_shader; 

    /* geometry */
    GLuint m_vbo, m_vao;
    /* compiled shaders */
    GLuint m_vs, m_fs;
    /* shader program */
    GLuint m_sp; 

   public:
      TexturedQuad();
      ~TexturedQuad();
      bool init();
      void draw();

};

#endif
