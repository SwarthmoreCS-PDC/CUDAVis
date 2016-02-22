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


#ifndef __BOOK_H__
#define __BOOK_H__

#include "handle_cuda_error.h"

/* Swap objects a and b using copy constructor */
template< typename T >
void swap( T& a, T& b ) {
    T t = a;
    a = b;
    b = t;
}


/* return a void* pointer to a buffer of length size bytes 
 * containing randomly generated data.
 * Each call to the function allocates and populates
 * a new random block. User assumes responsibility for freeing the
 * data block */
void* big_random_block( int size );

/* return a int* pointer to a buffer of length size integers 
 * containing randomly generated data.
 * Each call to the function allocates and populates
 * a new random block. User assumes responsibility for freeing the
 * data block */
int* big_random_block_int( int size );


// TODO: What are these?
__device__ unsigned char value( float n1, float n2, int hue );

__global__ void float_to_color( unsigned char *optr, const float *outSrc );

__global__ void float_to_color( uchar4 *optr, const float *outSrc );


/* Determine appropriate thread type for WIN32/OTHER */
#if _WIN32
//Windows threads.
#include <windows.h>

typedef HANDLE CUTThread;
typedef unsigned (WINAPI *CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC unsigned WINAPI
#define  CUT_THREADEND return 0

#else
//POSIX threads.
#include <pthread.h>

typedef pthread_t CUTThread;
typedef void *(*CUT_THREADROUTINE)(void *);

#define CUT_THREADPROC void
#define  CUT_THREADEND
#endif

//Create thread.
CUTThread start_thread( CUT_THREADROUTINE, void *data );

//Wait for thread to finish.
void end_thread( CUTThread thread );

//Destroy thread.
void destroy_thread( CUTThread thread );

//Wait for multiple threads.
void wait_for_threads( const CUTThread *threads, int num );

#endif  // __BOOK_H__
