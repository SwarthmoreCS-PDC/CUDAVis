#ifndef TIMER_CPU_H
#define TIMER_CPU_H

#include "timerBase.h"
#include <sys/time.h>

/* Time events on CPU side */
class CPUTimer: public TimerBase {
	private:
		//store start/stop times
		struct timeval startTime, stopTime, stopDiff, curDiff;
		//subtract before time from after and store in result
		void timeSub(struct timeval& result, 
				struct timeval before, struct timeval after);
	public:
		CPUTimer(); //Create a timer
		~CPUTimer(){/*do nothing*/}
		void start(); 
		void stop();
		void print(); 
		float elapsed(); 
};

#endif
