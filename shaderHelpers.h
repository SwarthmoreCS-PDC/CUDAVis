/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
#pragma once

#include <GL/glew.h>
#include <string>

typedef struct shaderProgramInfo_t {
  GLuint vertex;
  GLuint fragment;
  GLuint program;
} shaderProgramInfo;

/* Make a new shader program given filenames containing
 * vertex shader and fragment shader sources.
 * Returns struct with program set to 0 if error,
 * or valid vertex shader, fragment shader, and program object ID if success
 */
shaderProgramInfo makeProgram(const std::string &vsFileName,
                              const std::string &fsFileName);

/* Release program and shader IDs */
void freeProgram(const shaderProgramInfo &pinfo);

/* Read contents of file in fname and return contents as
 * string. Returns empty string if error */
std::string readFile(const std::string &fname);

/* Make a new shader of a given type using source code provided
 * in src. Returns 0 if error, or valid shader object ID if success
 * type:
 *   GL_VERTEX_SHADER, GL_FRAGMENT_SHADE, GL_GEOMETRY_SHADER,
 *   GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, or GL_COMPUTE_SHADER
 */
GLuint makeShader(const std::string &src, GLenum type);
