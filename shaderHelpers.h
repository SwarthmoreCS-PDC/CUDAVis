#ifndef SHADER_HELPERS_H
#define SHADER_HELPERS_H

#include <string>
#include <GL/glew.h>

/* Read contents of file in fname and return contents as
 * string. Returns empty string if error */
std::string readFile(const std::string& fname);

/* Make a new shader of a given type using source code provided
 * in src. Returns 0 if error, or valid shader object ID if success
 * type: 
 *   GL_VERTEX_SHADER, GL_FRAGMENT_SHADE, GL_GEOMETRY_SHADER,
 *   GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, or GL_COMPUTE_SHADER 
 */
GLuint makeShader(const std::string& src, GLenum type);

#endif
