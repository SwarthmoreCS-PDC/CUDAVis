#version 400

in vec2 vp;

void main(){
  gl_Position = vec4(vp, 0., 1.);
}

