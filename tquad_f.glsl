#version 400

out vec4 clr;

void main(){
  vec2 pos = gl_FragCoord.xy/512;
  clr = vec4(pos.x, 0.0, pos.y, 1.0);
}

