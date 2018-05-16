/* Copyright 2016-2018
 * Swarthmore College Computer Science, Swarthmore PA
 * T. Newhall, A. Danner
 */
#pragma once

#include <sys/time.h>

class TimerBase {
public:
  virtual ~TimerBase() { /*do nothing*/
  }
  virtual void start() = 0; // Start a timer
  virtual void stop() = 0;  // Stop timer
  // Print time in milliseconds between last start and stop
  virtual void print() = 0;
  // Get elapsed time from last start until now in milliseconds
  virtual float elapsed() = 0;
};
