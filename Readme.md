# An OpenGL Library for CUDA Visualization

This code allows provides an easy to use [GPUDisplayData](gpuDisplayData.h) class that helps users visualize data processed by CUDA kernels. Tested on CUDA 8.0 and OpenGL 4.0, the code should work on modern NVIDIA graphics hardware/software combinations.

# Requirements

  * An CUDA capable NVIDIA graphics card (See [caveats](#caveats) below)
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
./userBuffer
./ripple
./julia
```

After creating the `build` directory and running `cmake` once, you should only need to run `make -j8` in the build directory if you modify the files.

# Writing Your Own Animation

To wirte your own animation, follow the general outline provided in the examples [ripple.cu](ripple.cu), [userBuffer.cu](userBuffer.cu), and [julia.cu](julia.cu). The primary step is to write an animation function, e.g.,

```C
static void animate_ripple(uchar3 *disp, void *mycudadata);
```

The first parameter, `disp`, is a color buffer created automatically by the library on the GPU. The second parameter, `mycudadata` is a pointer to any user defined struct that provides additional information. The animation function will call a CUDA kernel to write values to the color buffer. After the kernel finishes, the library will display the image to the screen and then call the animation function again in a loop. Due to design limitation of the freeglut library, the animation function needs to be declared `static`.

The `main` function simply needs to create a `  GPUDisplayData` object specifying the dimensions of the desired image, a pointer to the user defined data, and a string title.

```C
GPUDisplayData my_display(info.size, info.size, &info, "Ripple CUDA");
```

To start the animation, call the `AnimateComputation` method with your new animation function, e.g.,

```C
my_display.AnimateComputation(animate_ripple);
```

If some code cleanup or post processing of user data needs to happen after closing the window, you can register a clean up function prior to starting the animation:

```C
my_display.RegisterExitFunction(clean_up);
```

In [userBuffer.cu](userBuffer.cu), the user specified data includes additional GPU buffers.

# Caveats

 CUDA began deprecating order compute architectures in CUDA 7.5. If you are using a new CUDA version with very old NVIDIA GPUs, the [CMakeLists.txt](CMakeLists.txt) may misconfigure your build and result in code that compiles but seems to display white noise instead of a useful image. You can comment out the line

 ```  
 LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30")
 ```

 To use the deprecated architectures, but these will likely be removed in future versions of CUDA. Additionally, more complex examples like [julia.cu](julia.cu) may run very slowly or not at all on older graphics hardware.  


# References

More details regarding how we use this project in courses at Swarthmore College in Computer Science can be found at our EduPar'18 site https://www.cs.swarthmore.edu/edupar18/

This library is inspired by and a simplified form of similar code from the text ["CUDA by Example: An Introduction to General-Purpose GPU Programming"](https://developer.nvidia.com/cuda-example). This software contains source code provided by NVIDIA Corporation. NVIDIA Copyright headers are retained in code copied verbatim from the book code samples.  
