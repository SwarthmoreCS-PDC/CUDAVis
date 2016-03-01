#version 400

in vec2 texUV;

out vec3 clr;

uniform sampler2D texSampler;

void main(){
  vec2 pos = gl_FragCoord.xy/512;
  vec3 tex = texture(texSampler, texUV).rgb;
  //clr = vec3(pos.x, 0.0, pos.y);
  //clr = vec3(texUV.x, 0.0, texUV.y);
  clr = tex;
}

