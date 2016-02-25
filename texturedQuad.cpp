#include <GL/glew.h>

#include "texturedQuad.h"

const float TexturedQuad::sm_points[] = { 
  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f };

const char* TexturedQuad::sm_vertex_shader = 
      "#version 400\n"
      "in vec2 vp;"
      "void main(){ gl_Position = vec4(vp, 0.0, 1.0);}";

const char* TexturedQuad::sm_fragment_shader = 
      "#version 400\n"
      "out vec4 clr;"
      "void main () { clr = vec4 (0.2, 0.0, 0.7, 1.0);}";


TexturedQuad::TexturedQuad(): 
  m_vbo(0), m_vao(0), m_vs(0), m_fs(0), m_sp(0) 
{ /*do nothing*/ }

bool TexturedQuad::init() {
  glGenBuffers(1, &m_vbo);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glBufferData (GL_ARRAY_BUFFER, 8*sizeof (float), sm_points, GL_STATIC_DRAW);

  glGenVertexArrays (1, &m_vao);
  glBindVertexArray (m_vao);
  glEnableVertexAttribArray(0);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  m_vs = glCreateShader (GL_VERTEX_SHADER);
  glShaderSource (m_vs, 1, &sm_vertex_shader, NULL);
  glCompileShader (m_vs);
  m_fs = glCreateShader (GL_FRAGMENT_SHADER);
  glShaderSource (m_fs, 1, &sm_fragment_shader, NULL);
  glCompileShader (m_fs);

  m_sp = glCreateProgram();
  glAttachShader (m_sp, m_fs);
  glAttachShader (m_sp, m_vs);
  glLinkProgram (m_sp);
  
  return true;
}


TexturedQuad::~TexturedQuad(){
  if(m_vs) { glDeleteShader(m_vs); }
  if(m_fs) { glDeleteShader(m_fs); }
  if(m_sp) { glDeleteProgram(m_sp);}
}

void TexturedQuad::draw(){
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glUseProgram (m_sp);
  glBindVertexArray (m_vao);
  // draw points 0-3 from the currently bound VAO with current in-use shader
  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
}


