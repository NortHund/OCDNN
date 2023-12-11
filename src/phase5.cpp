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

double* matrixL0sum;
double* matrixL1insum;
double* matrixL1outsum;

double* matrixL2sum;
double* matrixL3insum;
double* matrixL3outsum;

double* matrixL4sum;
double* matrixL5insum;
double* matrixL5outsum;

double* matrixW01double;
double* matrixW12double;
double* matrixW23double;
double* matrixW34double;
double* matrixW45double;
double* matrixW56double;

double* matrixW01sum;
double* matrixW23sum;
double* matrixW23csum;
double* matrixW45sum;

double* matrixB01double;
double* matrixB12double;
double* matrixB23double;
double* matrixB34double;
double* matrixB45double;
double* matrixB56double;

double *ics;
double* ocs;
double* csc;

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
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/p5-conv64.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_double"); //0
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "input_sum"); //1
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "output_sum"); //2
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "cs_compare"); //3
    }

    unsigned convolution_double(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *)&iBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *)&wBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *)&oBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 3, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 4, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 5, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 6, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 7, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 8, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 9, sizeof(int), &iln);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 10, sizeof(int), &olm);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(0),
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
        iBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * iw * ih * sizeof(double),
                                  iptr,
                                  NULL);

        wBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * od * ww * wh * sizeof(double),
                                  wptr,
                                  NULL);

        oBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  od * ow * oh * sizeof(double),
                                  NULL,
                                  NULL);
    }

    double convolution_double_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            oBuffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned input_sum(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(1), 0, sizeof(cl_mem), (void *)&iBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 1, sizeof(cl_mem), (void *)&icsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 2, sizeof(int), &id);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(1),
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

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned input_sum_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* iptr)
    {
        iBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * iw * ih * sizeof(double),
                                  iptr,
                                  NULL);


        icsBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  od * ow * oh * sizeof(double),
                                  NULL,
                                  NULL);
    }

    double input_sum_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            icsBuffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[3] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned output_sum(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&oBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&ocsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 2, sizeof(int), &id);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(2),
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

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned output_sum_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* iptr)
    {
        oBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  id * iw * ih * sizeof(double),
                                  iptr,
                                  NULL);


        ocsBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  od * ow * oh * sizeof(double),
                                  NULL,
                                  NULL);
    }

    double output_sum_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            ocsBuffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned cs_compare(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&icsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&ocsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(3),
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

        kernel_execution_times[6] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned cs_compare_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* iptr, double* optr)
    {
        icsBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 iw * ih * sizeof(double),
                                 iptr,
                                 NULL);


        ocsBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   ow * oh * sizeof(double),
                                   optr,
                                   NULL);

        cscBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE,
                                   5 * sizeof(double),
                                   NULL,
                                   NULL);
    }

    double cs_compare_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        csc = (double*)malloc(5 * sizeof(double));

        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     cscBuffer,
                                     0,
                                     0,
                                     5 * sizeof(double),
                                     csc,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[7] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        /*for (int i = 0; i < layer1h; i++) {
            for (int j = 0; j < layer1w; j++) {
                printf("csc:%f ",csc[i * layer1w + j]);
            }
            printf("\n");
        }
        printf("\n");
        printf("----------------------------\n");*/

        printf("csc: \n");
        for (int i = 0; i < 5; i++) {
            printf("%f ", csc[i]);
        }
        printf("\n");

        free(csc);

    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times:\n";
        std::cout << "  Convolution: " << kernel_execution_times[0] << " us\n";
        std::cout << "  Convolution read: " << kernel_execution_times[1] << " us\n";
        std::cout << "  Output sum: " << kernel_execution_times[4] << " us\n";
        std::cout << "  Output sum read: " << kernel_execution_times[5] << " us\n";
        std::cout << "  Checksum compare: " << kernel_execution_times[6] << " us\n";
        std::cout << "  Checksum compare read: " << kernel_execution_times[7] << " us\n";
        std::cout << "\n\n";
    }

    std::unique_ptr<OCL_Base> _ocl_base;

    cl_mem iBuffer = nullptr;
    cl_mem wBuffer = nullptr;
    cl_mem oBuffer = nullptr;
    cl_mem ocsBuffer = nullptr;
    cl_mem icsBuffer = nullptr;
    cl_mem cscBuffer = nullptr;

private:
    cl_program prog_cv_d;

    cl_event _event;

    // 0 - convolution
    // 1 - convolution read
    // 2 - input_sum
    // 3 - input_sum_read
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

    matrixL0sum = (double*)malloc((layer0w * layer0h) * sizeof(double));
    matrixL1insum = (double*)malloc((layer1w * layer1h) * sizeof(double));
    matrixL1outsum = (double*)malloc((layer1w * layer1h) * sizeof(double));
    matrixW01sum = (double*)malloc((layer0d) * (w01w * w01h) * sizeof(double));

    matrixL2sum = (double*)malloc((layer2w * layer2h) * sizeof(double));
    matrixL3insum = (double*)malloc((layer3w * layer3h) * sizeof(double));
    matrixL3outsum = (double*)malloc((layer3w * layer3h) * sizeof(double));
    matrixW23sum = (double*)malloc((layer2d) * (w23w * w23h) * sizeof(double));
    matrixW23csum = (double*)malloc((w23w * w23h) * sizeof(double));

    matrixL4sum = (double*)malloc((layer4w * layer4h) * sizeof(double));
    matrixL5insum = (double*)malloc((layer5w * layer5h) * sizeof(double));
    matrixL5outsum = (double*)malloc((layer5w * layer5h) * sizeof(double));
    matrixW45sum = (double*)malloc((layer4d) * (w45w * w45h) * sizeof(double));

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

    //matrixW01sum
    for (int x0 = 0; x0 < layer0d; ++x0) {
        for (int x2 = 0; x2 < w01h; ++x2) {
            for (int x3 = 0; x3 < w01h; ++x3) {
                matrixW01sum[(x0 * w01h * w01w) + (x2 * w01w) + x3] = 0;
                for (int x1 = 0; x1 < layer1d; ++x1) {
                    matrixW01sum[(x0 * w01h * w01w) + (x2 * w01w) + x3] += matrixW01double[
                            (x0 * layer1d * w01h * w01h) + (x1 * w01h * w01w) + (x2 * w01w) + x3];
                }
            }
        }
    }

    //matrixW23double
    for (int x0 = 0; x0 < layer2d; ++x0)
        for (int x1 = 0; x1 < layer3d; ++x1)
            for (int x2 = 0; x2 < w01h; ++x2)
                for (int x3 = 0; x3 < w01h; ++x3)
                    matrixW23double[(x0 * layer3d * w01h * w01w) + (x1 * w01h * w01w) + (x2 * w01w) +
                                    x3] = lenet->weight2_3[x0][x1][x2][x3];

    //matrixW23sum
    for (int x0 = 0; x0 < layer2d; ++x0) {
        for (int x2 = 0; x2 < w23h; ++x2) {
            for (int x3 = 0; x3 < w23h; ++x3) {
                matrixW23sum[(x0 * w23h * w23w) + (x2 * w23w) + x3] = 0;
                for (int x1 = 0; x1 < layer3d; ++x1) {
                    matrixW23sum[(x0 * w01h * w01w) + (x2 * w01w) + x3] += matrixW23double[
                            (x0 * layer3d * w23h * w23h) + (x1 * w23h * w23w) + (x2 * w23w) + x3];
                }
            }
        }
    }

    /*
    //matrixW23csum
    for (int x2 = 0; x2 < w23h; ++x2) {
        for (int x3 = 0; x3 < w23h; ++x3) {
            matrixW23csum[(x2 * w23w) + x3] = 0;
            for (int x0 = 0; x0 < layer2d; ++x0) {
                for (int x1 = 0; x1 < layer3d; ++x1) {
                    matrixW23csum[(x2 * w01w) + x3] += matrixW23double[
                            (x0 * layer3d * w23h * w23h) + (x1 * w23h * w23w) + (x2 * w23w) + x3];
                }
            }
        }
    }*/


    //matrixW45double
    for (int x0 = 0; x0 < layer4d; ++x0)
        for (int x1 = 0; x1 < layer5d; ++x1)
            for (int x2 = 0; x2 < w01h; ++x2)
                for (int x3 = 0; x3 < w01h; ++x3)
                    matrixW45double[(x0 * layer5d * w01h * w01h) + (x1 * w01h * w01w) + (x2 * w01w) +
                                    x3] = lenet->weight4_5[x0][x1][x2][x3];

    //matrixW45sum
    for (int x0 = 0; x0 < layer4d; ++x0) {
        for (int x2 = 0; x2 < w45h; ++x2) {
            for (int x3 = 0; x3 < w45h; ++x3) {
                matrixW45sum[(x0 * w45h * w45w) + (x2 * w45w) + x3] = 0;
                for (int x1 = 0; x1 < layer5d; ++x1) {
                    matrixW45sum[(x0 * w45h * w45w) + (x2 * w45w) + x3] += matrixW45double[
                            (x0 * layer5d * w45h * w45h) + (x1 * w45h * w45w) + (x2 * w45w) + x3];
                }
            }
        }
    }
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

    //input sum in c++
    /*for (int i = 0; i < layer0h; i++) {
        for (int j = 0; j < layer0w; j++) {
            matrixL0sum[i * layer0w + j] = matrixL0double[i * layer0w + j];
        }
    }*/

    //input sum matrix
    /*ocl_phase2.input_sum_write(layer0w, layer0h, 1, w01w, w01h, layer0w, layer0h, 1, 0, 0,
                                        matrixL0double);
    ocl_phase2.input_sum(layer0w, layer0h, 1, w01w, w01h, layer0w, layer0h, 1, 0, 0);
    ocl_phase2.input_sum_read(layer0w, layer0h, 1, w01w, w01h, layer0w, layer0h, 1, 0, 0,
                                       matrixL0sum);
    */

    /*for (int i = 0; i < layer0h; i++) {
        for (int j = 0; j < layer0w; j++) {
            printf("s:%f v:%f ",matrixL0sum[i * layer0w + j], matrixL0double[i * layer0w + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("----------------------------\n");*/

    //matrix cs:
    int counter = 0;
    ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, 1, 0, 0,
                                        matrixL0double,
                                        matrixW01sum);
    ocl_phase2.convolution_double(layer0w, layer0h, 1, w01w, w01h, layer1w, layer1h, 1, 0, 0);
    counter++;
    ocl_phase2.convolution_double_read(layer0w, layer0h, 1, w01w, w01h, layer1w, layer1h, 1, 0, 0,
                                       matrixL1insum);
    printf("conv layer 1 ics convolutions: %d \n", counter);

    ocl_phase2.print_kernel_execution_times();

    //ocl_phase2.print_kernel_execution_times();

    /*for (int i = 0; i < layer1h; i++) {
        for (int j = 0; j < layer1w; j++) {
            printf("%f ",matrixL1insum[i * layer1w + j]);
        }
        printf("\n");
    }*/


    //layer 1 convolution ocl
    ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0,
                                        matrixL0double,
                                        matrixW01double);

    //convolution
    counter = 0;
    for (int x = 0; x < (layer0d); ++x) {
        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, x, y);
            counter++;
        }
    }
    ocl_phase2.convolution_double_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0,
                                       matrixL1double);

    printf("conv layer 0-1 convolutions: %d \n", counter);
    ocl_phase2.print_kernel_execution_times();

    ocl_phase2.output_sum_write(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, 1, 0, 0,
                               matrixL1double);
    ocl_phase2.output_sum(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, 1, 0, 0);
    ocl_phase2.output_sum_read(layer1w, layer0h, layer1d, w01w, w01h, layer1w, layer1h, 1, 0, 0,
                              matrixL1outsum);

    //printf("\n");
    //printf("\n");
    for (int i = 0; i < layer1h; i++) {
        for (int j = 0; j < layer1w; j++) {
            if (abs(matrixL1insum[i * layer1w + j] - matrixL1outsum[i * layer1w + j]) > 0.0000001) {
                printf("checksum mismatch: in:%f out:%f ",matrixL1insum[i * layer1w + j] ,matrixL1outsum[i * layer1w + j]);
                printf("\n");
            }
            //printf("%f ",matrixL1insum[i * layer1w + j]);
        }
        //printf("\n");
    }
    //printf("\n");
    //printf("\n");
    ocl_phase2.cs_compare_write(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, matrixL1insum, matrixL1outsum);
    ocl_phase2.cs_compare(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
    ocl_phase2.cs_compare_read(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, csc);

    /* //compare in c++
    double check = 0;
    for (int i = 0; i < layer1h; i++) {
        for (int j = 0; j < layer1w; j++) {
            check = 0;
            for (int k = 0; k < layer1d; k++) {
                check += matrixL1double[(k * layer1w * layer1h) + (i * layer1w) + j];
            }
            //printf("s: %f , c: %f ",matrixL1sum[i * layer1w + j], check);
        }
        //printf("\n");
    }
    //printf("\n");*/

    //ocs
    ocs[0] = 0;

    /*printf("matrixW01sum: \n");
    for (int x0 = 0; x0 < layer0d; ++x0) {
        for (int x2 = 0; x2 < w01h; ++x2) {
            for (int x3 = 0; x3 < w01h; ++x3) {
                //printf("%f ", matrixW01sum[(x0 * w01h * w01w) + (x2 * w01w) + x3]);
            }
            //printf("\n");
        }
        //printf("\n");
    }*/

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


    //input sum matrix layer2
    /*ocl_phase2.input_sum_write(layer2w, layer2h, layer2d, w01w, w01h, layer2w, layer2h, 1, 0, 0,
                               matrixL2double);
    ocl_phase2.input_sum(layer2w, layer2h, layer2d, w01w, w01h, layer2w, layer2h, 1, 0, 0);
    ocl_phase2.input_sum_read(layer2w, layer2h, layer2d, w01w, w01h, layer2w, layer2h, 1, 0, 0,
                              matrixL2sum);*/

    /*printf("layer2\n");
    for (int x = 0; x < layer2d; ++x) {
        for (int j = 0; j < layer2h; ++j) {
            for (int i = 0; i < layer2w; ++i) {
                printf("%f ", matrixL2double[(x * layer2h * layer2w) + (j * layer2w) + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
    */

    /*
    printf("layer2sum:\n");
    for (int i = 0; i < layer2h; i++) {
        for (int j = 0; j < layer2w; j++) {
            printf("%f ",matrixL2sum[i * layer2w + j]);
        }
        printf("\n");
    }*/

    /*
    printf("matrixW23sum: \n");
    for (int x0 = 0; x0 < layer2d; ++x0) {
        for (int x2 = 0; x2 < w01h; ++x2) {
            for (int x3 = 0; x3 < w01h; ++x3) {
                printf("%f ", matrixW23sum[(x0 * w01h * w01w) + (x2 * w01w) + x3]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */

    /*
    printf("matrixW23csum: \n");
    for (int x0 = 0; x0 < 1; ++x0) {
        for (int x2 = 0; x2 < w01h; ++x2) {
            for (int x3 = 0; x3 < w01h; ++x3) {
                printf("%f ", matrixW23csum[(x2 * w01w) + x3]);
            }
            printf("\n");
        }
        printf("\n");
    }*/


    //layer 3 matrix cs:
    counter=0;
    ocl_phase2.convolution_double_write(layer2w, layer2h, layer2d, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                                        matrixL2double,
                                        matrixW23sum);
    for (int x = 0; x < (layer2d); ++x) {
        ocl_phase2.convolution_double(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, x, 0);
        counter++;
    }
    ocl_phase2.convolution_double_read(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                                       matrixL3insum);

    printf("conv layer 3 ics convolutions: %d \n", counter);
    ocl_phase2.print_kernel_execution_times();

    /*printf("layer3insum:\n");
    for (int i = 0; i < layer3h; i++) {
        for (int j = 0; j < layer3w; j++) {
            printf("%f ",matrixL3insum[i * layer3w + j]);
            matrixL3insum[i * layer3w + j] = 0;
        }
        printf("\n");
    }*/

    ocl_phase2.convolution_double_write(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                                        matrixL2sum,
                                        matrixW23sum);
        ocl_phase2.convolution_double(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, 0, 0);
    ocl_phase2.convolution_double_read(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                                       matrixL3insum);

    /*
    printf("layer3insum - compact:\n");
    for (int i = 0; i < layer3h; i++) {
        for (int j = 0; j < layer3w; j++) {
            printf("%f ",matrixL3insum[i * layer3w + j]);
        }
        printf("\n");
    }*/

    //layer3 convolution - working <3
    ocl_phase2.convolution_double_write(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0,
                                        matrixL2double,
                                        matrixW23double);

    counter=0;
    for (int x = 0; x < (layer2d); ++x) {
        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.convolution_double(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, x, y);
            counter++;
        }
    }
    //ocl_phase2.convolution_double(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0);
    ocl_phase2.convolution_double_read(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, 0, 0,
                                       matrixL3double);
    printf("conv layer 2-3 convolutions: %d \n",counter);
    ocl_phase2.print_kernel_execution_times();

    ocl_phase2.output_sum_write(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                                matrixL3double);
    ocl_phase2.output_sum(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, 1, 0, 0);
    ocl_phase2.output_sum_read(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, 1, 0, 0,
                               matrixL3outsum);

    //printf("layer3outsum:\n");
    for (int i = 0; i < layer3h; i++) {
        for (int j = 0; j < layer3w; j++) {
            //printf("%f ",matrixL3outsum[i * layer3w + j]);
        }
        //printf("\n");
    }

    ocl_phase2.cs_compare_write(layer3w, layer3h, 1, w01w, w01h, layer3w, layer3h, layer3d, 0, 0, matrixL3insum, matrixL3outsum);
    ocl_phase2.cs_compare(layer3w, layer3h, 1, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    ocl_phase2.cs_compare_read(layer3w, layer3h, 1, w01w, w01h, layer3w, layer3h, layer3d, 0, 0, csc);

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

    //input sum matrix layer4
    /*ocl_phase2.input_sum_write(layer4w, layer4h, layer4d, w01w, w01h, layer4w, layer4h, 1, 0, 0,
                               matrixL4double);
    ocl_phase2.input_sum(layer4w, layer4h, layer4d, w01w, w01h, layer4w, layer4h, 1, 0, 0);
    ocl_phase2.input_sum_read(layer4w, layer4h, layer4d, w01w, w01h, layer4w, layer4h, 1, 0, 0,
                              matrixL4sum);*/

    //layer 5 matrix cs:
    ocl_phase2.convolution_double_write(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, 1, 0, 0,
                                        matrixL4double,
                                        matrixW45sum);
    counter =0;
    for (int x = 0; x < (layer4d); ++x) {
        ocl_phase2.convolution_double(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, 1, x, 0);
        counter++;
    }
    ocl_phase2.convolution_double_read(layer4w, layer4h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0,
                                       matrixL5insum);

    printf("conv layer 5 ics convolutions: %d \n",counter);
    ocl_phase2.print_kernel_execution_times();

    //layer 5 convolution ocl
    ocl_phase2.convolution_double_write(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0,
                                        matrixL4double,
                                        matrixW45double);

    counter = 0;
    for (int x = 0; x < (layer4d); ++x) {
        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.convolution_double(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, x, y);
            counter++;
        }
    }
    ocl_phase2.convolution_double_read(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0,
                                       matrixL5double);
    printf("conv layer 4-5 convolutions: %d \n",counter);
    ocl_phase2.print_kernel_execution_times();

    //layer 5 output cs
    ocl_phase2.output_sum_write(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, 1, 0, 0,
                                matrixL5double);
    ocl_phase2.output_sum(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, 1, 0, 0);
    ocl_phase2.output_sum_read(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, 1, 0, 0,
                               matrixL5outsum);


    ocl_phase2.cs_compare_write(layer5w, layer5h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0, matrixL5insum, matrixL5outsum);
    ocl_phase2.cs_compare(layer5w, layer5h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0);
    ocl_phase2.cs_compare_read(layer5w, layer5h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0, csc);

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
    printf("layer3: original: %f, ocl accelerated: %f\n", ocs4, ocs5);

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
    printf("layer5: original: %f, ocl accelerated: %f\n", ocs8, ocs9);

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

    free(matrixL0sum);
    free(matrixL1insum);
    free(matrixL1outsum);

    free(matrixL2sum);
    free(matrixL3insum);
    free(matrixL3outsum);

    free(matrixL4sum);
    free(matrixL5insum);
    free(matrixL5outsum);

    free(matrixW01double);
    free(matrixW12double);
    free(matrixW23double);
    free(matrixW34double);
    free(matrixW45double);
    free(matrixW56double);

    free(matrixW01sum);
    free(matrixW23sum);
    free(matrixW23csum);
    free(matrixW45sum);

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
