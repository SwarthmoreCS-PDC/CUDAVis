/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */

#version 400

layout(location = 0) in vec2 vp;
layout(location = 1) in vec2 texCoord;

out vec2 texUV;

void main() {
  gl_Position = vec4(vp, 0., 1.);
  texUV = texCoord;
}
