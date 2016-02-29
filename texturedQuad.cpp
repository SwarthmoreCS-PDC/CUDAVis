#include <iostream>
#include <GL/glew.h>

#include "texturedQuad.h"
#include "shaderHelpers.h"

const float TexturedQuad::sm_points[] = { 
  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f };

TexturedQuad::TexturedQuad(): 
  m_vbo(0), m_vao(0)
{ m_pinfo.vertex = m_pinfo.fragment = m_pinfo.program = 0; }
  

bool TexturedQuad::init(const std::string& vshader,
                        const std::string& fshader) {
  glGenBuffers(1, &m_vbo);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glBufferData (GL_ARRAY_BUFFER, 8*sizeof (float), sm_points, GL_STATIC_DRAW);

  glGenVertexArrays (1, &m_vao);
  glBindVertexArray (m_vao);
  glEnableVertexAttribArray(0);
  //glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  m_pinfo = makeProgram(vshader, fshader);
  if(!m_pinfo.program){ return false; }
 
  return true;
}


TexturedQuad::~TexturedQuad(){
  freeProgram(m_pinfo);
}

void TexturedQuad::draw(){
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram (m_pinfo.program);
  glBindVertexArray (m_vao);
  // draw points 0-3 from the currently bound VAO with current in-use shader
  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
}


