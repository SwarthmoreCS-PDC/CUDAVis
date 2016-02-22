#include "timer.h"
#include <cstdio>

#define HANDLE_ERROR(a) a

using namespace std;

CPUTimer::CPUTimer() {
    gettimeofday(&startTime,NULL);
    gettimeofday(&stopTime,NULL);
}

void CPUTimer::start() {
    gettimeofday(&startTime,NULL);
}

void CPUTimer::stop() {
    gettimeofday(&stopTime,NULL);
    timeSub(stopDiff, startTime, stopTime);
}

void CPUTimer::timeSub(
        struct timeval& result, struct timeval before,  struct timeval after){

    if(after.tv_usec < before.tv_usec){
        after.tv_sec -= 1;
        after.tv_usec += 1000000;
    }
    result.tv_sec = after.tv_sec - before.tv_sec;
    result.tv_usec = after.tv_usec - before.tv_usec;
}

void CPUTimer::print() {
    float diff = stopDiff.tv_sec+stopDiff.tv_usec/1000000.0;
    printf("Elapsed time: %.3f \n", diff);
}

float CPUTimer::elapsed() {
    struct timeval now;
    gettimeofday(&now,NULL);
    timeSub(curDiff, startTime, now);
    return curDiff.tv_sec+curDiff.tv_usec/(1000.*1000);
}


GPUTimer::GPUTimer() {
    HANDLE_ERROR( cudaEventCreate(&startTime) );
    HANDLE_ERROR( cudaEventCreate(&stopTime) );
}

GPUTimer::~GPUTimer(){
    HANDLE_ERROR( cudaEventDestroy(startTime) );
    HANDLE_ERROR( cudaEventDestroy(stopTime) );
}

void GPUTimer::start() {
    HANDLE_ERROR( cudaEventRecord(startTime,0) );
}

void GPUTimer::stop() {
    HANDLE_ERROR( cudaEventRecord(stopTime,0) );
    HANDLE_ERROR( cudaEventSynchronize(stopTime) );
}

void GPUTimer::print() {
    printf("GPU Time: %7.2f ms\n", this->elapsed());
}

float GPUTimer::elapsed() {
    float elapsed_ms;
    this->stop();
    HANDLE_ERROR( cudaEventElapsedTime(&elapsed_ms, startTime, stopTime) );
    return elapsed_ms;
}
