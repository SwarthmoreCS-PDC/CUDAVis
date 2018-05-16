/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
#include "timerCPU.h"
#include <cstdio>

CPUTimer::CPUTimer() {
  gettimeofday(&startTime, NULL);
  gettimeofday(&stopTime, NULL);
}

void CPUTimer::start() { gettimeofday(&startTime, NULL); }

void CPUTimer::stop() {
  gettimeofday(&stopTime, NULL);
  timeSub(stopDiff, startTime, stopTime);
}

void CPUTimer::timeSub(struct timeval &result, struct timeval before,
                       struct timeval after) {

  if (after.tv_usec < before.tv_usec) {
    after.tv_sec -= 1;
    after.tv_usec += 1000000;
  }
  result.tv_sec = after.tv_sec - before.tv_sec;
  result.tv_usec = after.tv_usec - before.tv_usec;
}

void CPUTimer::print() {
  float diff = stopDiff.tv_sec * 1000 + stopDiff.tv_usec / 1000.0;
  printf("Elapsed time: %.3f \n", diff);
}

float CPUTimer::elapsed() {
  struct timeval now;
  gettimeofday(&now, NULL);
  timeSub(curDiff, startTime, now);
  return curDiff.tv_sec * 1000 + curDiff.tv_usec / (1000.);
}
