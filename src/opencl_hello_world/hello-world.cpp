/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following 
conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the 
following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following 
disclaimer in the documentation and/or  other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, 
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include "hello-world.h"

OCL_HelloWorld::OCL_HelloWorld()
{
}

OCL_HelloWorld::~OCL_HelloWorld()
{
    /*Step 12: Clean the resources.*/
    status = clReleaseMemObject(inputBuffer); //Release mem object.
    status = clReleaseMemObject(outputBuffer);

    if (output != NULL)
    {
        free(output);
        output = NULL;
    }
}

void OCL_HelloWorld::Run()
{
    /*Step 7: Initial input,output for the host and create memory objects for the kernel*/
    input = "GdkknVnqkc";
    size_t strlength = strlen(input);
    std::cout << "input string:" << std::endl;
    std::cout << input << std::endl;
    output = (char*)malloc(strlength + 1);

    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             (strlength + 1) * sizeof(char), (void *)input, NULL);
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 
                              (strlength + 1) * sizeof(char), NULL, NULL);

    cl_program prog = CreateProgramFromFile("hello-world.cl");
    CreateKernelFromProgram(prog, "helloworld");

    /*Step 9: Sets Kernel arguments.*/
    status = clSetKernelArg(GetKernel(0), 0, sizeof(cl_mem), (void *)&inputBuffer);
    status = clSetKernelArg(GetKernel(0), 1, sizeof(cl_mem), (void *)&outputBuffer);

    /*Step 10: Running the kernel.*/
    size_t global_work_size[1] = { strlength };
    status = clEnqueueNDRangeKernel(commandQueue, GetKernel(0), 1, NULL, 
                                        global_work_size, NULL, 0, NULL, NULL);

    /*Step 11: Read the std::cout put back to host memory.*/
    status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, 
                 strlength * sizeof(char), output, 0, NULL, NULL);

    output[strlength] = '\0'; //Add the terminal character to the end of output.
    std::cout << "\noutput string:" << std::endl;
    std::cout << output << std::endl;

}