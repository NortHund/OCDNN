#include "Stopwatches.h"
#include <chrono>

int ChronoClock::now()
{
    return std::chrono::duration_cast<std::chrono::microseconds>
            (std::chrono::system_clock::now().time_since_epoch()).count();
}


Stopwatch::Stopwatch(IClock& clock) 
    : mClock{clock}
{
    ;
}

int Stopwatch::saveStartPoint()
{
    return mStartTime = mClock.now();
}

int Stopwatch::saveEndPoint()
{
    return mEndTime = mClock.now();
}

int Stopwatch::getElapsedTime()
{
    if (mStartTime < mEndTime)
    {
        return mEndTime - mStartTime;
    }
    else
    {
        return 0;
    }
}


ProgramStopwatch::ProgramStopwatch(IClock& clock) 
    : Stopwatch{clock}
    , mClock{clock}
{
    ;
}

int ProgramStopwatch::runProgram(IProgram& program)
{
    saveStartPoint();
    int result = program.run();
    saveEndPoint();
    return result;
}
