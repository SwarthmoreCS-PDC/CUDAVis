#include <iostream>
#include <GL/glew.h>

#include "texturedQuad.h"
#include "shaderHelpers.h"

const float TexturedQuad::sm_points[] = { 
  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f };

TexturedQuad::TexturedQuad(): 
  m_vbo(0), m_vao(0), m_vs(0), m_fs(0), m_sp(0) 
{ /*do nothing*/ }

bool TexturedQuad::loadShader(const std::string& fname, GLenum type){
  GLuint id;
  std::string shaderSource = readFile(fname);
  if(shaderSource.empty()){
    std::cerr << "Error reading shader source: " << fname << std::endl;
    return false;
  }

  id = makeShader(shaderSource, type);
  if(id == 0){
    std::cerr << "Error compiling shader: " << fname << std::endl;
    return false;
  }

  switch(type){
    case GL_VERTEX_SHADER:
      m_vs = id; break;
    case GL_FRAGMENT_SHADER:
      m_fs = id; break;
    default:
      std::cerr << "Unknown shader type " << type << std::endl;
  }

  return true;
}
  

bool TexturedQuad::init(const std::string& vshader,
                        const std::string& fshader) {
  glGenBuffers(1, &m_vbo);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glBufferData (GL_ARRAY_BUFFER, 8*sizeof (float), sm_points, GL_STATIC_DRAW);

  glGenVertexArrays (1, &m_vao);
  glBindVertexArray (m_vao);
  glEnableVertexAttribArray(0);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo);
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

  if (! loadShader(vshader, GL_VERTEX_SHADER) ) { return false; }
  if (! loadShader(fshader, GL_FRAGMENT_SHADER) ) { return false; }
  //m_vs = glCreateShader (GL_VERTEX_SHADER);
  //glShaderSource (m_vs, 1, &sm_vertex_shader, NULL);
  //glCompileShader (m_vs);
  //m_fs = glCreateShader (GL_FRAGMENT_SHADER);
  //glShaderSource (m_fs, 1, &sm_fragment_shader, NULL);
  //glCompileShader (m_fs);

  m_sp = glCreateProgram();
  if( !m_sp ){ return false; }

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


