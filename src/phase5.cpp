#include "phase5.h"

int layer0w = 32;
int layer0h = 32;
int layer0d = 1;

int layer1w = 28;
int layer1h = 28;
int layer1d = 6;

int layer2w = LENGTH_FEATURE2;
int layer2h = LENGTH_FEATURE2;
int layer2d = LAYER2;

int layer3w = LENGTH_FEATURE3;
int layer3h = LENGTH_FEATURE3;
int layer3d = LAYER3;

int layer4w = LENGTH_FEATURE4;
int layer4h = LENGTH_FEATURE4;
int layer4d = LAYER4;

int layer5w = LENGTH_FEATURE5;
int layer5h = LENGTH_FEATURE5;
int layer5d = LAYER5;

int w01w = 5;
int w01h = 5;

int w12w = 5;
int w12h = 5;

int w23w = 5;
int w23h = 5;

int w34w = 5;
int w34h = 5;

int w45w = 5;
int w45h = 5;

int w56w = 5;
int w56h = 5;

double* matrixL0double;
double* matrixL1double;
double* matrixL2double;
double* matrixL3double;
double* matrixL4double;
double* matrixL5double;
double* matrixL6double;

double* matrixW01double;
double* matrixW12double;
double* matrixW23double;
double* matrixW34double;
double* matrixW45double;
double* matrixW56double;

double* matrixB01double;
double* matrixB12double;
double* matrixB23double;
double* matrixB34double;
double* matrixB45double;
double* matrixB56double;

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
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/p5-conv64.cl");
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
        _ocl_base->CreateKernelFromProgram(prog_cv_d , "convolution_optim2_ics"); //13
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
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            std::cerr << "At d: " <<  iln << " d2: " << olm  << std::endl;
            exit(EXIT_FAILURE);
        }

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution_double_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* iptr, double* wptr)
    {
        iBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * iw * ih * sizeof(double),
                                  iptr,
                                  NULL);

        wBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * od * ww * wh * sizeof(double),
                                  wptr,
                                  NULL);

        oBufferd = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  od * ow * oh * sizeof(double),
                                  NULL,
                                  NULL);
    }

    double convolution_double_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            oBufferd,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
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

    unsigned convolution_optimv2_ics(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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
        status = clSetKernelArg(_ocl_base->GetKernel(13), 0, sizeof(cl_mem), (void *)&iBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 1, sizeof(cl_mem), (void *)&wBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 2, sizeof(cl_mem), (void *)&icsBuffer);
        //status = clSetKernelArg(_ocl_base->GetKernel(13), 3, sizeof(cl_mem), (void *)&midRWBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 3, k1 * 2 * sizeof(double), NULL);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 4, k1 * 2 * sizeof(double), NULL);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 5, (k1 * 2) * (k1 * 2) * sizeof(double), NULL);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 6, wh * wh * sizeof(double), NULL);
        //status = clSetKernelArg(_ocl_base->GetKernel(13), 4, sizeof(cl_mem), (void *)&midCLBuffer);
        //status = clSetKernelArg(_ocl_base->GetKernel(13), 5, sizeof(cl_mem), (void *)&cornerMatBuffer);
        //status = clSetKernelArg(_ocl_base->GetKernel(13), 6, sizeof(cl_mem), (void *)&matSumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 7, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 8, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 9, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 10, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 11, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 12, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 13, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(13), 14, sizeof(int), &od);

        size_t global_work_size[1];
        global_work_size[0] = 1;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(13),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[8] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(12), 0, sizeof(cl_mem), (void *)&oBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 1, sizeof(cl_mem), (void *)&ocsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 2, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 3, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 4, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(12), 5, sizeof(int), &ocsInd);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(12),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[7] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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
        std::cout << "  Input checksum optimized v2: " << kernel_execution_times[8] << " us\n";
        std::cout << "  Input checksum read: " << kernel_execution_times[3] << " us\n";
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
    unsigned long kernel_execution_times[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

OCL_Phase2 ocl_phase2;

static void createVectors()
{
    matrixL0double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL1double = (double*)malloc((layer1d) * (layer1w * layer1h) * sizeof(double));
    matrixL2double = (double*)malloc((layer2d) * (layer2w * layer2h) * sizeof(double));
    matrixL3double = (double*)malloc((layer3d) * (layer3w * layer3h) * sizeof(double));
    matrixL4double = (double*)malloc((layer4d) * (layer4w * layer4h) * sizeof(double));
    matrixL5double = (double*)malloc((layer5d) * (layer5w * layer5h) * sizeof(double));
    matrixL6double = (double*)malloc(OUTPUT * sizeof(double));

    matrixW01double = (double*)malloc((layer0d) * (layer1d) * (w01w * w01h) * sizeof(double));
    matrixW12double = (double*)malloc((layer1d) * (layer2d) * (w12w * w12h) * sizeof(double));
    matrixW23double = (double*)malloc((layer2d) * (layer3d) * (w23w * w23h) * sizeof(double));
    matrixW34double = (double*)malloc((layer3d) * (layer4d) * (w34w * w34h) * sizeof(double));
    matrixW45double = (double*)malloc((layer4d) * (layer5d) * (w45w * w45h) * sizeof(double));
    matrixW56double = (double*)malloc((layer5d) * (OUTPUT) * (layer5w * layer5h) * sizeof(double));

    matrixB01double = (double*)malloc((layer1d) * sizeof(double));
    matrixB12double = (double*)malloc((layer2d) * sizeof(double));
    matrixB23double = (double*)malloc((layer3d) * sizeof(double));
    matrixB34double = (double*)malloc((layer4d) * sizeof(double));
    matrixB45double = (double*)malloc((layer5d) * sizeof(double));
    matrixB56double = (double*)malloc(OUTPUT * sizeof(double));

    for (int i = 0; i < layer0h; i++) {
        for (int j = 0; j < layer0w; j++) {
            matrixL0double[i * layer0w + j] = 0;
        }
    }

    ics = (double*)malloc(sizeof(double));
    ocs = (double*)malloc(sizeof(double));
}

static void copyModel(LeNet5 *lenet) {
    //matrixW01double
    for (int x0 = 0; x0 < layer0d; ++x0)
        for (int x1 = 0; x1 < layer1d; ++x1)
            for (int x2 = 0; x2 < w01h; ++x2)
                for (int x3 = 0; x3 < w01h; ++x3)
                    matrixW01double[(x0 * layer1d * w01h * w01h) + (x1 * w01h * w01w) + (x2 * w01w) + x3] = lenet->weight0_1[x0][x1][x2][x3];

    //matrixW23double
    for (int x0 = 0; x0 < layer2d; ++x0)
        for (int x1 = 0; x1 < layer3d; ++x1)
            for (int x2 = 0; x2 < w01h; ++x2)
                for (int x3 = 0; x3 < w01h; ++x3)
                    matrixW23double[(x0 * layer3d * w01h * w01w) + (x1 * w01h * w01w) + (x2 * w01w) +
                                    x3] = lenet->weight2_3[x0][x1][x2][x3];

    //matrixW45double
    for (int x0 = 0; x0 < layer4d; ++x0)
        for (int x1 = 0; x1 < layer5d; ++x1)
            for (int x2 = 0; x2 < w01h; ++x2)
                for (int x3 = 0; x3 < w01h; ++x3)
                    matrixW45double[(x0 * layer5d * w01h * w01h) + (x1 * w01h * w01w) + (x2 * w01w) +
                                    x3] = lenet->weight4_5[x0][x1][x2][x3];
}

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define FOREACH(i,count) for (int i = 0; i < count; ++i)

#define CONVOLUTE_VALID(input,output,weight)											\
{																						\
	FOREACH(o0,GETLENGTH(output))														\
		FOREACH(o1,GETLENGTH(*(output)))												\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];	\
}

#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
					(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];	\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)					\
{																				\
	for (int x = 0; x < GETLENGTH(weight); ++x)									\
		for (int y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[x][y]);                 \
	FOREACH(j, GETLENGTH(output))												\
		FOREACH(i, GETCOUNT(output[j]))											\
		((double *)output[j])[i] = action(((double *)output[j])[i] + bias[j]);	\
}

#define SUBSAMP_MAX_FORWARD(input,output)														\
{																								\
	const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));								\
	const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));								\
	FOREACH(i, GETLENGTH(output))																\
	FOREACH(o0, GETLENGTH(*(output)))															\
	FOREACH(o1, GETLENGTH(**(output)))															\
	{																							\
		int x0 = 0, x1 = 0, ismax;																\
		FOREACH(l0, len0)																		\
			FOREACH(l1, len1)																	\
		{																						\
			ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
			x0 += ismax * (l0 - x0);															\
			x1 += ismax * (l1 - x1);															\
		}																						\
		output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];								\
	}																							\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	for (int x = 0; x < GETLENGTH(weight); ++x)								\
		for (int y = 0; y < GETLENGTH(*weight); ++y)						\
			((double *)output)[y] += ((double *)input)[x] * weight[x][y];	\
	FOREACH(j, GETLENGTH(bias))												\
		((double *)output)[j] = action(((double *)output)[j] + bias[j]);	\
} \

double relu(double x)
{
    return x*(x > 0);
}

double relugrad(double y)
{
    return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
    CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
    SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
    CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
    SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
    CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
    DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void forward_ocl(LeNet5 *lenet, double* inptr, double* weightptr, double* outptr, double* biasptr, int ind, int inh, int inw, int outd, int outh, int outw, int wmh, int wmw)
{
    ocl_phase2.convolution_double_write(inw, inh, ind, wmw, wmh, outw, outh, outd, 0, 0,
                                        inptr,
                                        weightptr);
    for (int x = 0; x < (ind); ++x) {
        for (int y = 0; y < outd; ++y) {
            ocl_phase2.convolution_double(inw, inh, ind, w01w, w01h, outw, outh, outd, x, y);
        }
    }
    ocl_phase2.convolution_double_read(inw, inh, ind, w01w, w01h, outw, outh, outd, 0, 0,
                                       outptr);

    //Relu
    for (int i=0;i<outd;++i) {
        for (int j=0;j<outh;++j) {
            for (int k=0;k<outw;++k) {
                if (outptr[(i * outh * outw) + (j * outw) + k] + biasptr[i] > 0) {
                    outptr[(i * outh * outw) + (j * outw) + k] += biasptr[i];
                } else {
                    outptr[(i * outh * outw) + (j * outw) + k] = 0;
                }
            }
        }
    }
}

static inline void load_input(Feature *features, image input)
{
    double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
        {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    mean /= sz;
    std = sqrt(std / sz - mean*mean);
    FOREACH(j, sizeof(image) / sizeof(*input))
        FOREACH(k, sizeof(*input) / sizeof(**input))
        {
            layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
        }
}

static inline void load_input_ocl(image input)
{
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;
    FOREACH(j, sizeof(image) / sizeof(*input))FOREACH(k, sizeof(*input) / sizeof(**input)) {
            mean += input[j][k];
            std += input[j][k] * input[j][k];
        }
    mean /= sz;
    std = sqrt(std / sz - mean * mean);
    double testIn = 0;
    for (int i = 0; i < layer1h; i++) {
        for (int j = 0; j < layer1w; j++) {
            testIn = (input[i][j] - mean) / std;
            matrixL0double[(i + PADDING) * layer0w + (j + PADDING)] = testIn;
        }
    }
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], int label, int count)
{
    double inner = 0;
    for (int i = 0; i < count; ++i)
    {
        double res = 0;
        for (int j = 0; j < count; ++j)
        {
            res += exp(input[j] - input[i]);
        }
        loss[i] = 1. / res;
        inner -= loss[i] * loss[i];
    }
    inner += loss[label];
    for (int i = 0; i < count; ++i)
    {
        loss[i] *= (i == label) - loss[i] - inner;
    }
}

static void load_target(Feature *features, Feature *errors, int label)
{
    double *output = (double *)features->output;
    double *error = (double *)errors->output;
    softmax(output, error, label, GETCOUNT(features->output));
}

static uint8 get_result(Feature *features, uint8 count)
{
    double *output = (double *)features->output;
    const int outlen = GETCOUNT(features->output);
    uint8 result = 0;
    double maxvalue = *output;
    for (uint8 i = 1; i < count; ++i)
    {
        if (output[i] > maxvalue)
        {
            maxvalue = output[i];
            result = i;
        }
    }
    return result;
}

static double f64rand()
{
    static int randbit = 0;
    if (!randbit)
    {
        srand((unsigned)time(0));
        for (int i = RAND_MAX; i; i >>= 1, ++randbit);
    }
    unsigned long long lvalue = 0x4000000000000000L;
    int i = 52 - randbit;
    for (; i > 0; i -= randbit)
        lvalue |= (unsigned long long)rand() << i;
    lvalue |= (unsigned long long)rand() >> -i;
    return *(double *)&lvalue - 3;
}


uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(lenet, &features, relu);
    return get_result(&features, count);
}

uint8 Predict_ocl(LeNet5 *lenet, image input,uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    //forward_ocl(lenet, );
    return get_result(&features, count);
}

void Initial(LeNet5 *lenet)
{
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = f64rand());
    for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
    for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
    for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
    for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
    for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}

#define FILE_TEST_IMAGE		"../../source-data/t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"../../source-data/t10k-labels-idx1-ubyte"
#define LENET_FILE 		"../../source-data/model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data)*count, 1, fp_image);
    fread(label,count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}


int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

int testing_ocl(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict_ocl(lenet, test_data[i], 10);
        right += l == p;
        if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);
    }
    return right;
}

int save(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "wb");
    if (!fp) return 1;
    fwrite(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int load(LeNet5 *lenet, char filename[])
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) return 1;
    fread(lenet, sizeof(LeNet5), 1, fp);
    fclose(fp);
    return 0;
}

int main() {
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    image *test_data = (image *) calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *) calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR!!!\nDataset File Not Found! Please Copy Dataset to the Folder Including the exe\n");
        free(test_data);
        free(test_label);
        system("pause");
    }


    LeNet5 *lenet = (LeNet5 *) malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);

    createVectors();
    //int right = testing(lenet, test_data, test_label, COUNT_TEST);
    //int right_ocl = testing_ocl(lenet, test_data, test_label, COUNT_TEST);

    //for (int i=0; i<2;i++) { //probably I run out of gpu memory if i is large, I'm creating new and new buffers
    //    int ip = Predict(lenet, test_data[i + 100], 10);
    //    int op = Predict_ocl(lenet, test_data[i + 100], 10);
    //    printf("ip: %d, op: %d\n", ip, op);
    //}

    copyModel(lenet);
    //load input image
    load_input_ocl(test_data[2]);

    /*//layer1 convolution c++ version
    for (int x = 0; x <(layer0d); ++x) {
        for (int y = 0; y < layer1d; ++y) {
            for (int row = 0; row < layer1h; ++row) {
                for (int col = 0; col < layer1w; ++col) {

                    double sum = 0;
                    for (int i = 0; i < w01w; i++) {
                        for (int j = 0; j < w01w; j++) {
                            sum += matrixL0double[(x * layer0h * layer0w) + ((row + i) * layer0w) + (col + j)] *
                                   matrixW01double[(x * layer1d * w01w * w01w) + (y * w01w * w01w) +
                                                   (i * w01w) + j];
                        }
                    }
                    matrixL1double[(y * layer1w * layer1h) + (row * layer1w) + col] = sum;
                }
            }
        }
    }*/

    //layer 1 convolution ocl
    ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0,
                                        matrixL0double,
                                        matrixW01double);

    //ics
    ics[0] = 0;
    ocl_phase2.convolution_double_ics_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
    ocl_phase2.convolution_double_ics(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d,0,0);
    ocl_phase2.convolution_double_ics_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

    //convolution
    for (int x = 0; x < (layer0d); ++x) {
        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, x, y);
        }
    }
    ocl_phase2.convolution_double_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0,
                                       matrixL1double);

    //ocs
    ocs[0] = 0;
    ocl_phase2.convolution_double_ocs_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
    printf("layer 1 ics ocl: %f ics ocl: %f \n", ics[0], ocs[0]);

    //Relu
    for (int i = 0; i < layer1d; ++i) {
        for (int j = 0; j < layer1h; ++j) {
            for (int k = 0; k < layer1w; ++k) {
                if (matrixL1double[(i * layer1h * layer1w) + (j * layer1w) + k] + lenet->bias0_1[i] > 0) {
                    matrixL1double[(i * layer1h * layer1w) + (j * layer1w) + k] += lenet->bias0_1[i];
                } else {
                    matrixL1double[(i * layer1h * layer1w) + (j * layer1w) + k] = 0;
                }
            }
        }
    }

    //layer 2 subsampling
    const int len0 = (layer1h / layer2h);
    const int len1 = (layer1w / layer2w);
    for (int i = 0; i < (layer2d); ++i)
        for (int o0 = 0; o0 < (layer2h); ++o0)
            for (int o1 = 0; o1 < (layer2w); ++o1) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len0; ++l0)
                    for (int l1 = 0; l1 < len1; ++l1) {
                        ismax = matrixL1double[((i) * layer1h * layer1w) + ((o0 * len0 + l0) * layer1w) +
                                               (o1 * len1 + l1)]
                                > matrixL1double[((i) * layer1h * layer1w) + ((o0 * len0 + x0) * layer1w) +
                                                 (o1 * len1 + x1)];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                matrixL2double[(i * layer2h * layer2w) + (o0 * layer2w) + o1] = matrixL1double[
                        ((i) * layer1h * layer1w) + ((o0 * len0 + x0) * layer1w) + (o1 * len1 + x1)];
            }


    //layer3 convolution - working <3
    ocl_phase2.convolution_double_write(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0,
                                        matrixL2double,
                                        matrixW23double);

    //ics
    ics[0] = 0;
    ocl_phase2.convolution_double_ics_write(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    ocl_phase2.convolution_double_ics(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d,0,0);
    ocl_phase2.convolution_double_ics_read(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);

    for (int x = 0; x < (layer2d); ++x) {
        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.convolution_double(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, x, y);
        }
    }
    //ocl_phase2.convolution_double(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0);
    ocl_phase2.convolution_double_read(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0,
                                       matrixL3double);

    //ocs
    ocs[0] = 0;
    ocl_phase2.convolution_double_ocs_write(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs_read(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0, 0);
    printf("layer 3 ics ocl: %f ics ocl: %f \n", ics[0], ocs[0]);

/*
    //zero out layer3 first before doing c++ conv:
    for (int y = 0; y < layer3d; ++y) {
        for (int row = 0; row < layer3h; ++row) {
            for (int col = 0; col < layer3w; ++col) {
                matrixL3double[(y * layer3w * layer3h) + (row * layer3w) + col] = 0;
            }
        }
    }
    //layer3 convolution in c++
    double ocssum = 0;
    for (int x = 0; x <(layer2d); ++x) {
        for (int y = 0; y < layer3d; ++y) {
            for (int row = 0; row < layer3h; ++row) {
                for (int col = 0; col < layer3w; ++col) {

                    double sum = matrixL3double[(y * layer3w * layer3h) + (row * layer3w) + col];
                    for (int i = 0; i < w01w; i++) {
                        for (int j = 0; j < w01w; j++) {
                            sum += matrixL2double[(x * layer2h * layer2w) + ((row + i) * layer2w) + (col + j)] *
                                   matrixW23double[(x * layer3d * w01w * w01w) + (y * w01w * w01w) +
                                          (i * w01w) + j];
                            ocssum += matrixL2double[(x * layer2h * layer2w) + ((row + i) * layer2w) + (col + j)] *
                                      matrixW23double[(x * layer3d * w01w * w01w) + (y * w01w * w01w) +
                                                     (i * w01w) + j];
                        }
                    }
                    matrixL3double[(y * layer3w * layer3h) + (row * layer3w) + col] = sum;
                }
            }
        }
    }*/

    //Relu
    for (int i = 0; i < layer3d; ++i) {
        for (int j = 0; j < layer3h; ++j) {
            for (int k = 0; k < layer3w; ++k) {
                if (matrixL3double[(i * layer3h * layer3w) + (j * layer3w) + k] + lenet->bias2_3[i] > 0) {
                    matrixL3double[(i * layer3h * layer3w) + (j * layer3w) + k] += lenet->bias2_3[i];
                } else {
                    matrixL3double[(i * layer3h * layer3w) + (j * layer3w) + k] = 0;
                }
            }
        }
    }

    // layer 4 subsampling
    const int len3 = (layer3h / layer4h);
    const int len4 = (layer3w / layer4w);
    for (int i = 0; i < (layer4d); ++i)
        for (int o0 = 0; o0 < (layer4h); ++o0)
            for (int o1 = 0; o1 < (layer4w); ++o1) {
                int x0 = 0, x1 = 0, ismax;
                for (int l0 = 0; l0 < len3; ++l0)
                    for (int l1 = 0; l1 < len4; ++l1) {
                        ismax = matrixL3double[((i) * layer3h * layer3w) + ((o0 * len3 + l0) * layer3w) +
                                               (o1 * len4 + l1)]
                                > matrixL3double[((i) * layer3h * layer3w) + ((o0 * len3 + x0) * layer3w) +
                                                 (o1 * len4 + x1)];
                        x0 += ismax * (l0 - x0);
                        x1 += ismax * (l1 - x1);
                    }
                matrixL4double[(i * layer4h * layer4w) + (o0 * layer4w) + o1] = matrixL3double[
                        ((i) * layer3h * layer3w) + ((o0 * len3 + x0) * layer3w) + (o1 * len4 + x1)];
            }

    //layer 5 convolution ocl
    ocl_phase2.convolution_double_write(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0,
                                        matrixL4double,
                                        matrixW45double);

    //ics
    ics[0] = 0;
    ocl_phase2.convolution_double_ics_write(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0);
    ocl_phase2.convolution_double_ics(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d,0,0);
    ocl_phase2.convolution_double_ics_read(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0);

    for (int x = 0; x < (layer4d); ++x) {
        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.convolution_double(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, x, y);
        }
    }
    ocl_phase2.convolution_double_read(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0,
                                       matrixL5double);

    //ocs
    ocs[0] = 0;
    ocl_phase2.convolution_double_ocs_write(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0, 0);
    ocl_phase2.convolution_double_ocs_read(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0, 0);
    printf("layer 5 ics ocl: %f ics ocl: %f \n", ics[0], ocs[0]);


    /*
    //zero out layer5 first before doing c++ conv:
    for (int y = 0; y < layer5d; ++y) {
        for (int row = 0; row < layer5h; ++row) {
            for (int col = 0; col < layer5w; ++col) {
                matrixL5double[(y * layer5w * layer5h) + (row * layer5w) + col] = 0;
            }
        }
    }
    //layer5 conv in c++
    for (int x = 0; x <(layer4d); ++x) {
        for (int y = 0; y < layer5d; ++y) {
            for (int row = 0; row < layer5h; ++row) {
                for (int col = 0; col < layer5w; ++col) {

                    double sum = matrixL5double[(y * layer5w * layer5h) + (row * layer5w) + col];
                    for (int i = 0; i < w01w; i++) {
                        for (int j = 0; j < w01w; j++) {
                            sum += matrixL4double[(x * layer4h * layer4w) + ((row + i) * layer4w) + (col + j)] *
                                   matrixW45double[(x * layer5d * w01w * w01w) + (y * w01w * w01w) +
                                                   (i * w01w) + j];
                        }
                    }
                    matrixL5double[(y * layer5w * layer5h) + (row * layer5w) + col] = sum;
                }
            }
        }
    }
    */

    //Relu
    for (int i = 0; i < layer5d; ++i) {
        for (int j = 0; j < layer5h; ++j) {
            for (int k = 0; k < layer5w; ++k) {
                if (matrixL5double[(i * layer5h * layer5w) + (j * layer5w) + k] + lenet->bias4_5[i] > 0) {
                    matrixL5double[(i * layer5h * layer5w) + (j * layer5w) + k] += lenet->bias4_5[i];
                } else {
                    matrixL5double[(i * layer5h * layer5w) + (j * layer5w) + k] = 0;
                }
            }
        }
    }

    //output layer matrix multiplication
    for (int y = 0; y < (sizeof(*lenet->weight5_6) / sizeof(*(*lenet->weight5_6))); ++y) {
        matrixL6double[y] = 0;
    }

    for (int x = 0; x < (sizeof(lenet->weight5_6) / sizeof(*(lenet->weight5_6))); ++x) {
        for (int y = 0; y < (sizeof(*lenet->weight5_6) / sizeof(*(*lenet->weight5_6))); ++y) {
            matrixL6double[y] += matrixL5double[x] * lenet->weight5_6[x][y];
        }
    }

    for (int j = 0; j < (sizeof(lenet->bias5_6) / sizeof(*(lenet->bias5_6))); ++j) {
        if (matrixL6double[j] + lenet->bias4_5[j] > 0) {
            matrixL6double[j] += lenet->bias4_5[j];
        } else {
            matrixL6double[j] = 0;
        }
    }

    //getting result from the output matrix/vector
    const int outlen = OUTPUT;
    uint8 result = 0;
    double maxvalue = 0;
    for (uint8 i = 1; i < OUTPUT; ++i) {
        if (matrixL6double[i] > maxvalue) {
            maxvalue = matrixL6double[i];
            result = i;
        }
    }

    //original version of Lenet for reference
    Feature features = {0};
    load_input(&features, test_data[2]);
    CONVOLUTION_FORWARD(features.input, features.layer1, lenet->weight0_1, lenet->bias0_1, relu);
    double ocs0 = 0;
    double ocs1 = 0;
    for (int x = 0; x < layer1d; ++x) {
        for (int j = 0; j < layer1w; ++j) {
            for (int i = 0; i < layer1h; ++i) {
                ocs0 += features.layer1[x][j][i];
                ocs1 += matrixL1double[(x * layer1h * layer1w) + (i * layer1w) + j];
            }
        }
    }
    printf("layer1: original: %f, accelerated: %f \n", ocs0, ocs1);

    /*ocs1 = 0;
    for (int i = 0; i < layer1d; ++i) {
        for (int j = 0; j < layer1h; ++j) {
            for (int k = 0; k < layer1w; ++k) {
                if (abs(features.layer1[i][j][k] - matrixL1double[(i * layer1h * layer1w) + (j * layer1w) + k]) >
                    0.0000001) {
                    printf("mismatch: %f and %f \n", features.layer1[i][j][k],
                           matrixL1double[(i * layer1h * layer1w) + (j * layer1w) + k]);
                }
            }
        }
    }
    for (int x = 0; x < (layer1d * layer1h * layer1w); ++x)
        ocs1 += matrixL1double[x];
    printf("ocs1: %f\n", ocs1);*/

    SUBSAMP_MAX_FORWARD(features.layer1, features.layer2);
    double ocs2 = 0, ocs3 = 0;
    for (int x = 0; x < layer2d; ++x) {
        for (int j = 0; j < layer2h; ++j) {
            for (int i = 0; i < layer2w; ++i) {
                ocs2 += features.layer2[x][j][i];
                ocs3 += matrixL2double[(x * layer2h * layer2w) + (j * layer2w) + i];
                //printf("id:%d, s:%f, v:%f ", i, features.layer2[x][j][i], matrixL2double[(x * layer2h * layer2w) + (j * layer2w) + i]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
    printf("layer2: original: %f, vectorized: %f\n", ocs2, ocs3);


    CONVOLUTION_FORWARD(features.layer2, features.layer3, lenet->weight2_3, lenet->bias2_3, relu);
    //matrixW23double
    for (int x0 = 0; x0 < layer2d; ++x0) {
        for (int x1 = 0; x1 < layer3d; ++x1) {
            for (int x2 = 0; x2 < w01h; ++x2) {
                for (int x3 = 0; x3 < w01h; ++x3) {
                    //printf("id %d, v: %f, s: %f ", x3, matrixW23double[(x0 * layer3d * w01h * w01h) + (x1 * w01h * w01w) + (x2 * w01w) + x3], lenet->weight2_3[x0][x1][x2][x3]);
                }
                //printf("\n");
            }
            //printf("\n");
        }
        //printf("------------------\n\n");  //why are the weights all wrong?
    }
    //printf("\n");
    //printf("\n");
    //printf("\n");

    double ocs4 = 0, ocs5 = 0;
    for (int x = 0; x < layer3d; ++x) {
        for (int j = 0; j < layer3h; ++j) {
            for (int i = 0; i < layer3w; ++i) {
                ocs4 += features.layer3[x][j][i];
                ocs5 += matrixL3double[(x * layer3h * layer3w) + (j * layer3w) + i];
                //printf("id:%d, s:%f, v:%f ", i, features.layer3[x][j][i], matrixL3double[(x * layer3h * layer3w) + (j * layer3w) + i]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
    printf("layer3: original: %f, vectorized: %f\n", ocs4, ocs5);

    SUBSAMP_MAX_FORWARD(features.layer3, features.layer4);
    double ocs6 = 0, ocs7 = 0;
    for (int x = 0; x < layer4d; ++x) {
        for (int j = 0; j < layer4h; ++j) {
            for (int i = 0; i < layer4w; ++i) {
                ocs6 += features.layer4[x][j][i];
                ocs7 += matrixL4double[(x * layer4h * layer4w) + (j * layer4w) + i];
                //printf("id:%d, s:%f, v:%f ", i, features.layer4[x][j][i], matrixL3double[(x * layer4h * layer4w) + (j * layer4w) + i]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
    printf("layer4: original: %f, vectorized: %f\n", ocs6, ocs7);

    CONVOLUTION_FORWARD(features.layer4, features.layer5, lenet->weight4_5, lenet->bias4_5, relu);
    double ocs8 = 0, ocs9 = 0;
    for (int x = 0; x < layer5d; ++x) {
        for (int j = 0; j < layer5h; ++j) {
            for (int i = 0; i < layer5w; ++i) {
                ocs8 += features.layer5[x][j][i];
                ocs9 += matrixL5double[(x * layer5h * layer5w) + (j * layer5w) + i];
                //printf("id:%d, s:%f, v:%f ", i, features.layer5[x][j][i], matrixL3double[(x * layer5h * layer5w) + (j * layer5w) + i]);
            }
            //printf("\n");
        }
        //printf("\n");
    }
    printf("layer5: original: %f, vectorized: %f\n", ocs8, ocs9);

    DOT_PRODUCT_FORWARD(features.layer5, features.output, lenet->weight5_6, lenet->bias5_6, relu);

    int prediction = get_result(&features, 10);



    printf("ocl prediction: %d true: %d\n", result, test_label[2]);
    printf("prediction: %d true: %d\n", prediction, test_label[2]);


    //printf("%d/%d\n", right, COUNT_TEST);
    //save(lenet, LENET_FILE);
    free(lenet);
    free(test_data);
    free(test_label);

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl_phase2.print_kernel_execution_times();

    free(matrixL0double);
    free(matrixL1double);
    free(matrixL2double);
    free(matrixL3double);
    free(matrixL4double);
    free(matrixL5double);
    free(matrixL6double);

    free(matrixW01double);
    free(matrixW12double);
    free(matrixW23double);
    free(matrixW34double);
    free(matrixW45double);
    free(matrixW56double);

    free(matrixB01double);
    free(matrixB12double);
    free(matrixB23double);
    free(matrixB34double);
    free(matrixB45double);
    free(matrixB56double);

    free(ics);
    free(ocs);

    printPlatformInfo(false);
    return 0;
}
