// Example code of measuring the time of execution of a program:
#include "Stopwatches.h"
#include <iostream>

class Program : public IProgram
{
    public:
    int run() override
    {
        int sum = 0;
        for (int i = 0; i < 1000000; ++i)
        {
            sum += i;
        }
        return sum;
    }
};

int main()
{

    ChronoClock clock;
    ProgramStopwatch programStopwatch(clock);
    Program program;

    int result = programStopwatch.runProgram(program);

    std::cout << "Result: " << result << std::endl;
    std::cout << "Elapsed time: " << programStopwatch.getElapsedTime() << " us" << std::endl;
}
