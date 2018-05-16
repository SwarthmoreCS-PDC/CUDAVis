/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
#include "shaderHelpers.h"
#include <fstream>
#include <iostream>
#include <sstream>

std::string readFile(const std::string &fname) {
  std::stringstream ss; /* default is empty */
  std::ifstream file;
  file.open(fname.c_str());
  if (!file) {
    return ss.str();
  }

  ss << file.rdbuf();
  file.close();
  return ss.str();
}

shaderProgramInfo makeProgram(const std::string &vsFileName,
                              const std::string &fsFileName) {

  shaderProgramInfo pinfo;
  GLuint id;
  std::string shaderSource;

  pinfo.program = 0;

  shaderSource = readFile(vsFileName);
  if (shaderSource.empty()) {
    std::cerr << "Error reading vertex shader source: " << vsFileName
              << std::endl;
    return pinfo;
  }

  id = makeShader(shaderSource, GL_VERTEX_SHADER);
  if (id == 0) {
    std::cerr << "Error compiling vertex shader: " << vsFileName << std::endl;
  } else {
    pinfo.vertex = id;
  }

  shaderSource = readFile(fsFileName);
  if (shaderSource.empty()) {
    std::cerr << "Error reading fragment shader source: " << fsFileName
              << std::endl;
    return pinfo;
  }

  id = makeShader(shaderSource, GL_FRAGMENT_SHADER);
  if (id == 0) {
    std::cerr << "Error compiling fragment shader: " << fsFileName << std::endl;
  } else {
    pinfo.fragment = id;
  }

  id = glCreateProgram();
  glAttachShader(id, pinfo.vertex);
  glAttachShader(id, pinfo.fragment);
  glLinkProgram(id);

  pinfo.program = id;
  return pinfo;
}

void freeProgram(const shaderProgramInfo &pinfo) {
  if (pinfo.vertex) {
    glDeleteShader(pinfo.vertex);
  }
  if (pinfo.fragment) {
    glDeleteShader(pinfo.fragment);
  }
  if (pinfo.program) {
    glDeleteProgram(pinfo.program);
  }
}

GLuint makeShader(const std::string &src, GLenum type) {
  const char *ptr = src.c_str();
  GLint ok;
  GLuint sh = glCreateShader(type);
  glShaderSource(sh, 1, &ptr, NULL);
  glCompileShader(sh);

  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (ok) {
    return sh;
  }

  /* if error, print it */
  GLint size = 0;   /* size of log */
  GLsizei read = 0; /* amount of info returned by log */
  glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &size);
  if (size > 1) {
    GLchar *compiler_log = new GLchar[size];
    glGetShaderInfoLog(sh, size, &read, compiler_log);
    std::cerr << "compiler_log:\n" << compiler_log << std::endl;
    delete[] compiler_log;
    compiler_log = NULL;
  }
  return 0;
}
