/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
#pragma once

#include "shaderHelpers.h"
#include <GL/glew.h>
#include <string>

class TexturedQuad {

private:
  static const float sm_points[];
  static const float sm_texcoords[];

  /* geometry */
  GLuint m_vbo_points, m_vbo_tex, m_vao;

  /* program, shader IDs */
  shaderProgramInfo m_pinfo;

  /* pixel buffer object, texture ID */
  GLuint m_pbo, m_tex;

  /* width and height of pixel buffer for texture */
  int m_width, m_height;

public:
  /* Create a new unit Textured quad in the xy plane
   * with a texture pixel buffer with dimensions
   * width by heighyt rgb pixels */
  TexturedQuad(int width, int height);
  ~TexturedQuad();

  /* Connect geometry to shader programs */
  bool init(const std::string &vshader, const std::string &fshader);

  /* Draw textured square using current shader program */
  void draw();

  /* Get ID of Pixel Buffer Object connected to Quad */
  inline GLuint getPBO() { return m_pbo; }
};
