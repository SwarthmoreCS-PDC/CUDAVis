# An OpenGL Library for CUDA Visualization

This code allows provides an easy to use [GPUDisplayData](myOpenGLlib.h) class that helps users visualize data processed by CUDA kernels. Tested on CUDA 8.0 and OpenGL 4.0, the code should work on modern NVIDIA graphics hardware/software combinations.

# Requirements

  * An CUDA capable NVIDIA graphics card
  * [CUDA libraries](https://developer.nvidia.com/cuda-downloads)
  * OpenGL
  * [freeglut](http://freeglut.sourceforge.net/)
  * [glew](http://glew.sourceforge.net/)
  * [cmake](https://cmake.org/)

  Many of these can be installed through your package manager, e.g., `apt-get, yum, port, brew`.

# Quick Build Directions

This project uses `cmake` to compile files in a separate `build` directory.

```
mkdir build
cd build
cmake ..
make -j8
./simple
./ripple
./julia
```

After creating the `build` directory and running `cmake` once, you should only need to run `make -j8` in the build directory if you modify the files.

# Writing Your Own Animation

To wirte your own animation, follow the general outline provided in the examples [ripple.cu](ripple.cu), [simple.cu](simple.cu), and [julia.cu](julia.cu). The primary step is to write an animation function, e.g.,

```C
static void animate_ripple(uchar3 *disp, void *mycudadata);
```



# References

More details regarding how we use this project in courses at Swarthmore College in Computer Science can be found at our EduPar'18 site https://www.cs.swarthmore.edu/edupar18/

This library is inspired by and a simplified form of similar code from the text ["CUDA by Example: An Introduction to General-Purpose GPU Programming"](https://developer.nvidia.com/cuda-example). This software contains source code provided by NVIDIA Corporation. NVIDIA Copyright headers are retained in code copied verbatim from the book code samples.  
