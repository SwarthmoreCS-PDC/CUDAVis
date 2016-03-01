#ifndef _TEXTUREDQUAD_H
#define _TEXTUREDQUAD_H

#include <GL/glew.h>
#include <string>
#include "shaderHelpers.h"

class TexturedQuad {

  private:
    static const float sm_points[];
    static const float sm_texcoords[];

    /* geometry */
    GLuint m_vbo_points, m_vbo_tex, m_vao;
    
    /* program, shader IDs */
    shaderProgramInfo m_pinfo;

    /* texture ID */
    GLuint m_tex; 

   public:
      TexturedQuad();
      ~TexturedQuad();
      bool init(const std::string& vshader, const std::string& fshader);
      void draw();

};

#endif
