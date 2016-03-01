#include <iostream>
#include <GL/glew.h>

#include "texturedQuad.h"
#include "shaderHelpers.h"

/*  order of square vertices
 *   1 -- 3
 *   |    |
 *   2 -- 4
 */
const float TexturedQuad::sm_points[] = { 
  -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f };

/* tex coords are in [0,1]x[0,1] origin in lower left */
const float TexturedQuad::sm_texcoords[] = { 
   0.0f, 1.0f,  0.0f,  0.0f, 1.0f, 1.0f, 1.0f,  0.0f };


TexturedQuad::TexturedQuad(int width, int height): 
  m_vbo_points(0), m_vbo_tex(0), m_vao(0),
  m_pbo(0), m_tex(0),
  m_width(width), m_height(height)
{ m_pinfo.vertex = m_pinfo.fragment = m_pinfo.program = 0; }
  

bool TexturedQuad::init(const std::string& vshader,
                        const std::string& fshader) {
  glGenBuffers(1, &m_vbo_points);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo_points);
  glBufferData (GL_ARRAY_BUFFER, 8*sizeof (float), sm_points, GL_STATIC_DRAW);

  glGenBuffers(1, &m_vbo_tex);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo_tex);
  glBufferData (GL_ARRAY_BUFFER, 8*sizeof (float),
      sm_texcoords, GL_STATIC_DRAW);

  glGenVertexArrays (1, &m_vao);
  glBindVertexArray (m_vao);
  /* vertex shader points in position 0 */
  glEnableVertexAttribArray(0);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo_points);
  glVertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 0, NULL); 
  /* vertex shader texture coords in position 1 */
  glEnableVertexAttribArray(1);
  glBindBuffer (GL_ARRAY_BUFFER, m_vbo_tex);
  glVertexAttribPointer (1, 2, GL_FLOAT, GL_FALSE, 0, NULL);


  /* create an OpenGL buffer for pixel texture data */
  glGenBuffers( 1, &m_pbo );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER, m_pbo );
  glBufferData( GL_PIXEL_UNPACK_BUFFER, m_width * m_height * 3, 
      NULL, GL_DYNAMIC_DRAW );

  /* make ID for holding texture color data */
  glGenTextures(1, &m_tex);
  glBindTexture(GL_TEXTURE_2D, m_tex);
  /* allocate space for texture, but don't read it in yet */
  glTexStorage2D(GL_TEXTURE_2D, 2, GL_RGB8, m_width, m_height);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  m_pinfo = makeProgram(vshader, fshader);
  if(!m_pinfo.program){ return false; }

  return true;
}


TexturedQuad::~TexturedQuad(){
  /* reset bindings */
  glBindBuffer(GL_ARRAY_BUFFER,0);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D,0);

  /* delete buffers */
  glDeleteBuffers(1, &m_vbo_points);
  glDeleteBuffers(1, &m_vbo_tex);
  glDeleteBuffers(1, &m_pbo);
  glDeleteVertexArrays(1, &m_vao);
  glDeleteTextures(1, &m_tex);

  /* free shaders, program */
  freeProgram(m_pinfo);
}

void TexturedQuad::draw(){
  glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindVertexArray (m_vao);
  glUseProgram(m_pinfo.program);
  /* find location of sampler in fragment shader */
  GLint samplerLoc = glGetUniformLocation(m_pinfo.program, "texSampler");
  /* tell sampler to sample from texture 0 */
  glUniform1i(samplerLoc, 0);
  /* set texture 0 to be current texture */
  glActiveTexture(GL_TEXTURE0);
  /* bind texture data from m_tex to current texture */
  glBindTexture(GL_TEXTURE_2D, m_tex);
  /* read from PIXEL_UNPACK_BUFFER */
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_width, m_height, 
      GL_RGB, GL_UNSIGNED_BYTE, NULL);
  // draw points 0-3 from the currently bound VAO with current in-use shader
  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);
}


