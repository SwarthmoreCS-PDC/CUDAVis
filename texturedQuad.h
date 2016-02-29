#ifndef _TEXTUREDQUAD_H
#define _TEXTUREDQUAD_H

#include <GL/gl.h>
#include <string>

class TexturedQuad {

  private:
    static const float sm_points[];
    /* geometry */
    GLuint m_vbo, m_vao;
    /* compiled shaders */
    GLuint m_vs, m_fs;
    /* shader program */
    GLuint m_sp; 
   
    bool loadShader(const std::string& fname, GLenum type);

   public:
      TexturedQuad();
      ~TexturedQuad();
      bool init(const std::string& vshader, const std::string& fshader);
      void draw();

};

#endif
