#pragma once

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */
#include <cuda.h>
#include <cstdio>

/* Check if return status (err) is not cudaSuccess. If a
 * real error occurred, print error string and exit program
 * with EXIT_FAILURE
 *
 * err : error code returned by cuda function
 * file: source file name of line containing HandleError call
 * line: line number in file containing HandleError call
 *
 * */
void HandleError(cudaError_t err, const char *file, int line);

/* Macro for automatically determining file name and line number */
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/* Macro for handling bad malloc */
#define HANDLE_NULL(a)                                                         \
  {                                                                            \
    if (a == NULL) {                                                           \
      printf("Host memory failed in %s at line %d\n", __FILE__, __LINE__);     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
