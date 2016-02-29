#ifndef TIMER_BASE_H
#define TIMER_BASE_H

#include <sys/time.h>

class TimerBase {
	public:
		virtual ~TimerBase(){ /*do nothing*/ }
		virtual void start() = 0;    //Start a timer
		virtual void stop() = 0;     //Stop timer
		//Print time in seconds between last start and stop
		virtual void print() = 0;    
		//Get elapsed time from last start until now in seconds
		virtual float elapsed() = 0;
};

#endif
