#pragma once

#include "timerBase.h"
#include "cuda.h"

/* Time events on GPU side
   See timerBase for method description
 */
class GPUTimer: public TimerBase {
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
