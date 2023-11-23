#include "phase3.h"

int layer0w = 32;
int layer0h = 32;
int layer0d = 1;

int layer1w = 28;
int layer1h = 28;
int layer1d = 6;

int w01w = 5;
int w01h = 5;

double* matrixL0double;
double* matrixW01double;
double* matrixL1double;

double *ics;
double* ocs;

// Get kernel execution time in microseconds
unsigned long get_kernel_execution_time(cl_event &event, cl_command_queue &command_queue)
{
    clFinish(command_queue);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    return (time_end - time_start) / 1000;
}

class OCL_Phase2
{
public:
    OCL_Phase2()
    {
        _ocl_base.reset(new OCL_Base());

        init_programs();
        init_kernels();
    }

    ~OCL_Phase2()
    {
    }

    void init_programs()
    {
        prog_mm_int = _ocl_base->CreateProgramFromFile("kernels/p2-mm-int32.cl");
        prog_mm_float = _ocl_base->CreateProgramFromFile("kernels/p2-mm-f32.cl");
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/p3-conv32.cl");
        prog_cv_oc = _ocl_base->CreateProgramFromFile("kernels/convolution-oc.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_mm_int, "mm_int"); //0
        _ocl_base->CreateKernelFromProgram(prog_mm_float, "mm_float"); //1
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_fl"); //2
        _ocl_base->CreateKernelFromProgram(prog_cv_oc, "convolution_oc"); //3
        _ocl_base->CreateKernelFromProgram(prog_mm_int , "mm_short"); //4
        _ocl_base->CreateKernelFromProgram(prog_mm_int , "mm_char"); //5
        _ocl_base->CreateKernelFromProgram(prog_mm_float, "mm_double"); //6
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_double"); //7
        _ocl_base->CreateKernelFromProgram(prog_mm_int , "mm_int_cs"); //8
        _ocl_base->CreateKernelFromProgram(prog_cv_d , "convolution_double_ocs"); //9
        _ocl_base->CreateKernelFromProgram(prog_cv_d , "convolution_double_ics"); //10
        _ocl_base->CreateKernelFromProgram(prog_cv_d , "convolution_optim_ics"); //11
        _ocl_base->CreateKernelFromProgram(prog_cv_d , "convolution_optim_ocs"); //12
    }

    unsigned convolution_double(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(7), 0, sizeof(cl_mem), (void *)&iBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 1, sizeof(cl_mem), (void *)&wBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 2, sizeof(cl_mem), (void *)&oBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 3, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 4, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 5, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 6, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 7, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 8, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 9, sizeof(int), &iln);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 10, sizeof(int), &olm);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(7),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution_double_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        iBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  iw * ih * sizeof(double),
                                  matrixL0double,
                                  NULL);

        wBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * od * ww * wh * sizeof(double),
                                  matrixW01double,
                                  NULL);

        oBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  od * ow * oh * sizeof(double),
                                  matrixL1double,
                                  NULL);
    }

    double convolution_double_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            oBufferd,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            matrixL1double,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned convolution_double_ics(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(10), 0, sizeof(cl_mem), (void *)&iBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 1, sizeof(cl_mem), (void *)&wBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 2, sizeof(cl_mem), (void *)&icsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 3, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 4, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 5, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 6, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 7, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 8, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 9, sizeof(int), &iln);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 10, sizeof(int), &olm);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(10),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return 0;
    }

    unsigned convolution_optim_ics(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od)
    {
        double* midRW;
        double* midCL;
        double* cornerMat;
        double* matSum;

        int k1 = ww - 1;

        midRW = (double *) malloc(k1 * 2 * sizeof(double));
        midCL = (double*)malloc(k1 * 2 * sizeof(double));
        cornerMat = (double*)malloc((k1 * 2) * (k1 * 2) * sizeof(double));
        matSum = (double*)malloc(ww * wh * sizeof(double));

        cl_mem midRWBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   k1 * 2 * sizeof(double),
                                   midRW,
                                   NULL);

        cl_mem midCLBuffer = clCreateBuffer(_ocl_base->context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            k1 * 2 * sizeof(double),
                                            midCL,
                                            NULL);

        cl_mem cornerMatBuffer = clCreateBuffer(_ocl_base->context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            (k1 * 2) * (k1 * 2) * sizeof(double),
                                            cornerMat,
                                            NULL);

        cl_mem matSumBuffer = clCreateBuffer(_ocl_base->context,
                                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                            ww * wh * sizeof(double),
                                            matSum,
                                            NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(11), 0, sizeof(cl_mem), (void *)&iBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 1, sizeof(cl_mem), (void *)&wBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 2, sizeof(cl_mem), (void *)&icsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 3, sizeof(cl_mem), (void *)&midRWBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 4, sizeof(cl_mem), (void *)&midCLBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 5, sizeof(cl_mem), (void *)&cornerMatBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 6, sizeof(cl_mem), (void *)&matSumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 7, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 8, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 9, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 10, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 11, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 12, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 13, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 14, sizeof(int), &od);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(11),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[6] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        free(midRW);
        free(midCL);
        free(cornerMat);
        free(matSum);


        return 0;
    }

    unsigned convolution_double_ics_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm) {

        icsBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(double),
                                   ics,
                                   NULL);

    }

    double convolution_double_ics_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm) {

        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            icsBuffer,
                                            0,
                                            0,
                                            sizeof(double),
                                            ics,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[3] = get_kernel_execution_time(_event, _ocl_base->commandQueue);
        return ics[0];
    }

    unsigned convolution_double_ocs(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(9), 0, sizeof(cl_mem), (void *)&oBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 1, sizeof(cl_mem), (void *)&ocsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 2, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 3, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 4, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 5, sizeof(int), &ocsInd);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(9),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution_optim_ocs(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {
        double* ocsOp;
        int LOCAL_SIZE = 32;
        size_t numWorkGroups = (ow * ow) / LOCAL_SIZE;
        size_t localWorkSize[1];
        localWorkSize[0] = LOCAL_SIZE;

        ocsOp = (double*)malloc(numWorkGroups * sizeof(double));

        cl_mem ocsOpBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   numWorkGroups * sizeof(double),
                                   ocs,
                                   NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(12), 0, sizeof(cl_mem), (void *)&oBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 1, sizeof(cl_mem), (void *)&ocsOpBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 2, LOCAL_SIZE* sizeof(double), NULL);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 3, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 4, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 5, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 6, sizeof(int), &ocsInd);

        size_t global_work_size[1];
        global_work_size[0] = ow * oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(12),
                                        1,
                                        NULL,
                                        global_work_size,
                                        localWorkSize,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[7] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //Reading result from GPU memory to main memory
        status;
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     ocsBuffer,
                                     0,
                                     0,
                                     numWorkGroups * sizeof(double),
                                     ocsOp,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        printf("ocsOp final: %f\n", ocsOp[0]);
        free(ocsOp);

        return (unsigned)status;
    }

    unsigned convolution_double_ocs_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {
        ocsBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(double),
                                   ocs,
                                   NULL);


    }

    double convolution_double_ocs_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {
        //Reading result from GPU memory to main memory
        cl_int status;
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     ocsBuffer,
                                     0,
                                     0,
                                     sizeof(double),
                                     ocs,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //printf("Convolution-double ocl ocs: %f \n", ocs[ocsInd]);
    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times:\n";
        std::cout << "  Convolution: " << kernel_execution_times[0] << " us\n";
        std::cout << "  Convolution read: " << kernel_execution_times[1] << " us\n";
        std::cout << "  Input checksum: " << kernel_execution_times[2] << " us\n";
        std::cout << "  Input checksum optimized: " << kernel_execution_times[6] << " us\n";
        std::cout << "  Input checksum read: " << kernel_execution_times[3] << " us\n";
        std::cout << "  Output checksum: " << kernel_execution_times[4] << " us\n";
        std::cout << "  Output checksum optimized: " << kernel_execution_times[7] << " us\n";
        std::cout << "  Output checksum read: " << kernel_execution_times[5] << " us\n\n";
    }

    std::unique_ptr<OCL_Base> _ocl_base;

    cl_mem iBufferd = nullptr;
    cl_mem wBufferd = nullptr;
    cl_mem oBufferd = nullptr;
    cl_mem ocsBuffer = nullptr;
    cl_mem icsBuffer = nullptr;

private:
    cl_program prog_mm_int ;
    cl_program prog_mm_float;
    cl_program prog_cv_d;
    cl_program prog_cv_oc;

    cl_event _event;

    // 0 - convolution
    // 1 - convolution read
    // 2 - ics
    // 3 - ics read
    // 4 - ocs
    // 5 - ocs read
    unsigned long kernel_execution_times[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

OCL_Phase2 ocl_phase2;

struct IcsOptimized : public IProgram
{
    int run() override
    {
        int k = w01w;
        int k1 = (w01w-1);
        int k2 = (w01w-2);

        double midSQ = 0;
        for (int n = 0; n < layer0d; n++) {
            for (int i = k1; i < layer0h - k1; i++) {
                for (int j = k1; j < layer0w - k1; j++) {
                    midSQ += matrixL0double[i * layer0w + j];
                }
            }
        }

        double midRW[k1 * 2];
        int id = 0;
        for (int i = 0; i < k1; i++) {
            midRW[id] = 0;
            for (int j = k1; j < layer0w - k1; j++) {
                midRW[id] += matrixL0double[i * layer0w + j];
            }
            id++;
        }
        for (int i = layer0h - k1; i < layer0h; i++) {
            midRW[id] = 0;
            for (int j = k1; j < layer0w - k1; j++) {
                midRW[id] += matrixL0double[i * layer0w + j];
            }
            id++;
        }

        double midCL[k1*2];
        id = 0;
        for (int i = 0; i < k1; i++) {
            midCL[id] = 0;
            for (int j = k1; j < layer0w - k1; j++) {
                midCL[id] += matrixL0double[j * layer0w + i];
            }
            id++;
        }
        for (int i = layer0h - k1; i < layer0h; i++) {
            midCL[id] = 0;
            for (int j = k1; j < layer0w - k1; j++) {
                midCL[id] += matrixL0double[j * layer0w + i];
            }
            id++;
        }

        double cornerMat[(k1 * 2) * (k1 * 2)];
        id = 0;
        for (int i = 0; i < k1; i++) {
            for (int j = 0; j < k1; j++) {
                cornerMat[id] = matrixL0double[j * layer0w + i];
                id++;
            }
            for (int j = layer0w - k1; j < layer0w; j++) {
                cornerMat[id] = matrixL0double[j * layer0w + i];
                id++;
            }
        }
        for (int i = layer0h - k1; i < layer0h; i++) {
            for (int j = 0; j < k1; j++) {
                cornerMat[id] = matrixL0double[j * layer0w + i];
                id++;
            }
            for (int j = layer0w - k1; j < layer0w; j++) {
                cornerMat[id] = matrixL0double[j * layer0w + i];
                id++;
            }
        }
        /*printf("corner mat: \n");
        for (int i = 0; i < (k1 * 2); i++) {
            for (int j = 0; j < (k1 * 2); j++) {
                printf("%f ", cornerMat[i * (k1 * 2) + j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("midSQ: %f, midRW: %f, midCL %f \n", midSQ, midRW[0], midCL[0]);*/

        //matsum loop version 1
        double matSum[k * k];
        /*for (int ci = 0; ci < k; ci++) {
            for (int cj = 0; cj < k; cj++) {
                matSum[ci * k + cj] = 0;
                matSum[ci * k + cj] += midSQ;
                for (int i = ci; i < ci + k1; i++) {
                    matSum[ci * k + cj] += midRW[i];
                    matSum[ci * k + cj] += midCL[i];
                    for (int j = cj; j < cj + k1; j++) {
                        matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + j];
                    }
                }
            }
        }
        printf("mat Sum v1: \n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                printf("%f ", matSum[i + j]);
            }
            printf("\n");
        }
        printf("\n");*/

        //matsum loop version 2
        //first value is calculated in full
        matSum[0] = 0;
        matSum[0] += midSQ;
        for (int i = 0; i < k1; i++) {
            matSum[0] += midRW[i];
            matSum[0] += midCL[i];
            for (int j = 0; j < k1; j++) {
                matSum[0] += cornerMat[i * (k1 * 2) + j];
            }
        }
        double prevSum;
        //second value onwards with this loop, value based on previous value
        for (int ci = 0; ci < k; ci++) {
            if (ci % 2 == 0) {
                for (int cj = 0; cj < k; cj++) {
                    if (cj + ci > 0) {
                        if (cj == 0) { //downwards shift, left edge
                            //printf("downwards, left edge\n");
                            matSum[ci * k + cj] = prevSum;
                            matSum[ci * k + cj] -= midRW[ci - 1];
                            matSum[ci * k + cj] += midRW[ci + k2];
                            for (int j = cj; j < cj + k1; j++) {
                                matSum[ci * k + cj] -= cornerMat[(ci - 1) * (k1 * 2) + j];
                                matSum[ci * k + cj] += cornerMat[(ci + k2) * (k1 * 2) + j];
                            }
                        } else { //left to right
                            //printf("left to right\n");
                            matSum[ci * k + cj] = prevSum;
                            matSum[ci * k + cj] -= midCL[cj - 1];
                            matSum[ci * k + cj] += midCL[cj + k2];
                            for (int i = ci; i < ci + k1; i++) {
                                matSum[ci * k + cj] -= cornerMat[i * (k1 * 2) + cj - 1];
                                matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + cj + k2];
                            }
                        }
                        prevSum = matSum[ci * k + cj];
                        //printf("id: %d\n", (ci * k + cj));
                        //printf("matSum: %f\n", matSum[ci * k + cj]);
                    } else {
                        prevSum = matSum[0];
                    }
                }
            } else {
                for (int cj = k1; cj >= 0; cj--) {
                    if (cj == k1) { //downwards shift, right edge
                        //printf("downwards, right edge\n");
                        matSum[ci * k + cj] = prevSum;
                        matSum[ci * k + cj] -= midRW[ci - 1];
                        matSum[ci * k + cj] += midRW[ci + k2];
                        for (int j = cj; j > cj - k1; j--) {
                            matSum[ci * k + cj] -= cornerMat[(ci - 1) * (k1 * 2) + j];
                            matSum[ci * k + cj] += cornerMat[(ci + k2) * (k1 * 2) + j];
                        }
                    } else { //right to left
                        //printf("right to left\n");
                        matSum[ci * k + cj] = prevSum;
                        matSum[ci * k + cj] += midCL[cj];
                        matSum[ci * k + cj] -= midCL[cj + k1];
                        for (int i = ci; i < ci + k1; i++) {
                            matSum[ci * k + cj] -= cornerMat[i * (k1 * 2) + cj + k1];
                            matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + cj];
                        }
                    }
                    prevSum = matSum[ci * k + cj];
                    //printf("id: %d\n", (ci * k + cj));
                    //printf("matSum: %f\n", matSum[ci * k + cj]);
                }
            }

        }
        /*printf("mat Sum v2: \n");
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                printf("id: %d, %f ",(i * k + j), matSum[i * k + j]);
            }
            printf("\n");
        }
        printf("\n");*/

        double wSum = 0;
        //double xSum = 0;
        double checksum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                wSum = 0;
                for (int m = 0; m < layer1d; ++m) {
                    wSum += matrixW01double[(m * w01h * w01w) + (i * w01w) + j];
                }
                checksum += wSum * matSum[(i * w01w) + j];

                /*xSum = 0;
                for (int r = 0; r < layer1h; ++r) {
                    for (int c = 0; c < layer1w; ++c) {
                        xSum += matrixL0double[((r + i) * layer0w) + (c + j)];
                    }
                }
                printf("xsum: %f\n", xSum);*/
            }
            //printf("\n");
        }
        //printf("checkusm: %f\n", checksum);

        unsigned error = 0;
        return (int) error;
    }
};

struct Ocs : public IProgram
{
    int run() override
    {
        double checksum = 0;
        for (int d = 0; d < layer1d; d++) {
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    checksum += matrixL1double[(d * layer1h * layer1w) + (i * layer1w) + j];
                }
            }
        }
        //printf("output checksum: %f \n", checksum);

        unsigned error = 0;
        return (int) error;
    }
};

struct CreateMatrices : public IProgram
{
    int run() override
    {
        ics = (double *) malloc(sizeof(double));
        ics[0] = 0;
        ocs = (double*)malloc(sizeof(double));
        ocs[0] = 0;

        matrixL0double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
        matrixW01double = (double*)malloc((layer0d) * (layer1d) * (w01w * w01h) * sizeof(double));
        matrixL1double = (double*)malloc((layer1d) * (layer1w * layer1h) * sizeof(double));

        for (int i=0; i<layer0h; i++) {
            for (int j=0; j<layer0w; j++) {
                matrixL0double[i * layer0w + j] = 2.5;
            }
        }

        for (int n = 0; n < layer0d; n++) {
            for (int d = 0; d < layer1d; d++) {
                for (int i = 0; i < w01h; i++) {
                    for (int j = 0; j < w01w; j++) {
                        matrixW01double[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + (i * w01w) + j] = 0.01 + 0.01 * d;
                    }
                }
            }
        }

        for (int d = 0; d < layer1d; d++) {
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    matrixL1double[(d * layer1h * layer1w) + (i * layer1w) + j] = d;
                }
            }
        }

        unsigned error = 0;
        return (int) error;
    }
};

struct MatrixAdditionOCL : public IProgram
{
    int run() override
    {
        ics[0] = 0;
        ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        ocl_phase2.convolution_double_ics_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_double_ics(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_double_ics_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        //printf("ics: %.16f \n", ics[0]);

        ics[0] = 0;
        ocl_phase2.convolution_double_ics_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_optim_ics(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d);
        ocl_phase2.convolution_double_ics_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        //printf("Optimized ics: %.16f \n", ics[0]);

        for (int i = 0; i < layer0d; i++) {
            for (int j = 0; j < layer1d; j++) {
                ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, i, j);
            }
        }
        ocl_phase2.convolution_double_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        ocl_phase2.convolution_double_ocs_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
        ocl_phase2.convolution_double_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
        ocl_phase2.convolution_double_ocs_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);

        ocl_phase2.convolution_optim_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);

        //printf("ocs %d: %f \n", i, ocs[i]);
        if (abs(ics[0] - ocs[0]) > 0.00000000001) {
            printf("checksum mismatch! ics: %.16f, ocs: %.16f \n", ics[0], ocs[0]);
        }
        //printf("ics: %.16f, ocs: %.16f. \n", ics[0], ocs[0]);


        unsigned error = 0;
        return (int) error;
    }
};

struct SaveMatrixOCL : public IProgram
{
    int run() override {

        FILE *fp = fopen("../../output-data/matrixL1d.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int d = 0; d < layer1d; d++) {
            fprintf(fp, "//M = %d \n", d);
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    fprintf(fp, "%f ", matrixL1double[(d * layer1h * layer1w) + i * layer1w + j]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matrixL0d.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < layer0h; i++) {
            for (int j = 0; j < layer0w; j++) {
                fprintf(fp, "%f ", matrixL0double[i * layer0w + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matrixW01d.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }
        for (int n = 0; n < layer0d; n++) {
            fprintf(fp, "//N = %d--------------------------------------------\n", n);
            for (int d = 0; d < layer1d; d++) {
                fprintf(fp, "//M = %d \n", d);
                for (int i = 0; i < w01h; i++) {
                    for (int j = 0; j < w01w; j++) {
                        fprintf(fp, "%f ", matrixW01double[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + i * w01w + j]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
        }
        fclose(fp);

        unsigned error = 0;
        return (int) error;
    }
};

int main()
{
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    //Program
    CreateMatrices createMatrices;
    MatrixAdditionOCL matrixAdditionOCL;
    SaveMatrixOCL saveMatrixOCL;

    IcsOptimized icsOptimized;
    Ocs ocsc;

    int result = 0;

    result = Program_sw.runProgram(createMatrices);
    std::cout << "Creation of matrices: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(icsOptimized);
    std::cout << "Ics optimized c++: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(matrixAdditionOCL);
    std::cout << "OpenCL matrix addition: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(ocsc);
    std::cout << "Ocs c++: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveMatrixOCL);
    std::cout << "OpenCL matrix save: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl_phase2.print_kernel_execution_times();

    free(matrixL0double);
    free(matrixW01double);
    free(matrixL1double);

    free(ics);
    free(ocs);

    printPlatformInfo(false);
    return 0;
}
