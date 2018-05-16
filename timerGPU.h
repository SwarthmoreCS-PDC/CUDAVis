/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */

#pragma once

#include <cuda.h>
#include "timerBase.h"

/* Time events on GPU side
   See timerBase for method description
 */
class GPUTimer : public TimerBase {
private:
  cudaEvent_t startTime, stopTime;

public:
  GPUTimer();
  ~GPUTimer();
  void start();
  void stop();
  void print();
  float elapsed();
};
