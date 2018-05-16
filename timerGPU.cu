/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */

#include "handle_cuda_error.h"
#include "timerGPU.h"
#include <cstdio>

using namespace std;

GPUTimer::GPUTimer() {
  HANDLE_ERROR(cudaEventCreate(&startTime));
  HANDLE_ERROR(cudaEventCreate(&stopTime));
}

GPUTimer::~GPUTimer() {
  HANDLE_ERROR(cudaEventDestroy(startTime));
  HANDLE_ERROR(cudaEventDestroy(stopTime));
}

void GPUTimer::start() { HANDLE_ERROR(cudaEventRecord(startTime, 0)); }

void GPUTimer::stop() {
  HANDLE_ERROR(cudaEventRecord(stopTime, 0));
  HANDLE_ERROR(cudaEventSynchronize(stopTime));
}

void GPUTimer::print() { printf("GPU Time: %7.2f s\n", this->elapsed()); }

float GPUTimer::elapsed() {
  float elapsed_ms;
  this->stop();
  HANDLE_ERROR(cudaEventElapsedTime(&elapsed_ms, startTime, stopTime));
  return elapsed_ms;
}
