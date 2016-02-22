#include "timer.h"
#include <cstdio>

using namespace std;

Timer::Timer() {
  gettimeofday(&startTime,NULL); 
  gettimeofday(&stopTime,NULL); 
}

void Timer::start() {
  gettimeofday(&startTime,NULL); 
}

void Timer::stop() {
  gettimeofday(&stopTime,NULL);
  timeSub(stopDiff, startTime, stopTime); 
}

void Timer::timeSub(
    struct timeval& result, struct timeval before,  struct timeval after){
   
  if(after.tv_usec < before.tv_usec){
    after.tv_sec -= 1;
    after.tv_usec += 1000000;
  }
  result.tv_sec = after.tv_sec - before.tv_sec;
  result.tv_usec = after.tv_usec - before.tv_usec;
}

void Timer::print() {
  float diff = stopDiff.tv_sec+stopDiff.tv_usec/1000000.0;
  printf("Elapsed time: %.3f \n", diff);
}

float Timer::elapsed() {
  struct timeval now;
  gettimeofday(&now,NULL);
  timeSub(curDiff, startTime, now);
  return curDiff.tv_sec+curDiff.tv_usec/(1000.*1000);
}
