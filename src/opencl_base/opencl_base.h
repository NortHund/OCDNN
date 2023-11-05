#ifndef OPENCL_BASE_H
#define OPENCL_BASE_H
#include <iostream>
#include <string>
#include <fstream>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/opencl.h>
#include <cstring>

class OCL_Base
{
public:
    OCL_Base();
    ~OCL_Base();

    cl_program CreateProgramFromFile(const char* filename);
    cl_kernel CreateKernelFromProgram(cl_program program, const char* kernelName);

    cl_kernel GetKernel(int index)
    {
        return Kernels[index];
    }

    cl_context context;
    cl_command_queue commandQueue;
    cl_command_queue commandQueue2;

protected:
    void Run();
    cl_int status;

private:
    void Init();

    cl_uint numPlatforms;
    cl_platform_id platform;
    cl_platform_id* platforms;

    cl_uint numDevices;
    cl_device_id* devices;

    cl_program* Programs;
    int ProgCount;

    cl_kernel* Kernels;
    int KernCount;

};

int convertToString(const char *filename, std::string& s);

    /* data */
#endif // OPENCL_BASE_H