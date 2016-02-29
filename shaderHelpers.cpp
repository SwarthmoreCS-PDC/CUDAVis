#include "shaderHelpers.h"
#include <sstream>
#include <fstream>

std::string readFile(const std::string& fname){
  std::stringstream ss; /* default is empty */
  std::ifstream file;
  file.open(fname.c_str());
  if(!file){
    return ss.str();
  }

  ss << file.rdbuf();
  file.close();
  return ss.str();
}

GLuint makeShader(const std::string& src, GLenum type){
   return 0;
}
