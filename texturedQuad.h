#ifndef _TEXTUREDQUAD_H
#define _TEXTUREDQUAD_H

#include <GL/glew.h>
#include <string>
#include "shaderHelpers.h"

class TexturedQuad {

  private:
    static const float sm_points[];
    /* geometry */
    GLuint m_vbo, m_vao;
    
    /* program, shader IDs */
    shaderProgramInfo m_pinfo;
   

   public:
      TexturedQuad();
      ~TexturedQuad();
      bool init(const std::string& vshader, const std::string& fshader);
      void draw();

};

#endif
