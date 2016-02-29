#include "shaderHelpers.h"
#include <sstream>
#include <fstream>
#include <iostream>

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
shaderProgramInfo makeProgram(
    const std::string& vsFileName, const std::string* fsFileName){
  shaderProgramInfo pinfo;
  pinfo.program = 0;
  return pinfo;
}

void freeProgram(const shaderProgramInfo& pinfo){
  return;
}

GLuint makeShader(const std::string& src, GLenum type){
   const char* ptr = src.c_str();
   GLint ok;
   GLuint sh = glCreateShader(type);
   glShaderSource(sh,1,&ptr, NULL);
   glCompileShader(sh);

   glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
   if(ok){ 
     return sh;
   }

   /* if error, print it */
   GLint size = 0;    /* size of log */ 	
   GLsizei read = 0;  /* amount of info returned by log */
   glGetShaderiv(sh, GL_INFO_LOG_LENGTH , &size);       
   if (size > 1)
   {
     GLchar* compiler_log = new GLchar[size];
     glGetShaderInfoLog(sh, size, &read, compiler_log);
     std::cerr << "compiler_log:\n" << compiler_log << std::endl;
     delete [] compiler_log; compiler_log=NULL;
   }
   return 0;

}
