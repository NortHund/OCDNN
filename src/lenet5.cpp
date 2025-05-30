#include "lenet5.h"

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

int layer6w = OUTPUT;
int layer6h = 1;
int layer6d = 1;

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

double* matrixL1insum;
double* matrixL1outsum;

double* matrixL3insum;
double* matrixL3outsum;

double* matrixL5insum;
double* matrixL5outsum;

double* matrixL6insum;
double* matrixL6outsum;

double* matrixW01double;
double* matrixW12double;
double* matrixW23double;
double* matrixW34double;
double* matrixW45double;
double* matrixW56double;

double* matrixW01sum;
double* matrixW23sum;
double* matrixW45sum;
double* matrixW56sum;

double* matrixB01double;
double* matrixB12double;
double* matrixB23double;
double* matrixB34double;
double* matrixB45double;
double* matrixB56double;

double *ics;
double* ocs;
double* csc;

int abft_err = 0;
int abft_err_ind = 0;

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

class OCL
{
public:
    OCL()
    {
        _ocl_base.reset(new OCL_Base());

        init_programs();
        init_kernels();
    }

    ~OCL()
    {
        free_bufs();
    }

    void init_programs()
    {
        prog_cv = _ocl_base->CreateProgramFromFile("kernels/lenet5-conv.cl");
        prog_util = _ocl_base->CreateProgramFromFile("kernels/lenet5-util.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_cv, "convolution_double"); //0
        _ocl_base->CreateKernelFromProgram(prog_cv, "input_sum"); //1
        _ocl_base->CreateKernelFromProgram(prog_cv, "output_sum"); //2
        _ocl_base->CreateKernelFromProgram(prog_cv, "cs_compare"); //3
        _ocl_base->CreateKernelFromProgram(prog_util, "relu"); //4
        _ocl_base->CreateKernelFromProgram(prog_util, "maxpool"); //5
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat"); //6
        _ocl_base->CreateKernelFromProgram(prog_util, "relu_d"); //7
        _ocl_base->CreateKernelFromProgram(prog_util, "maxpool_d"); //8
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat_ics"); //9
        _ocl_base->CreateKernelFromProgram(prog_cv, "convolution_ic"); //10
        _ocl_base->CreateKernelFromProgram(prog_cv, "output_r"); //11
    }

    double buf_read(int ow, int oh, int od, double* optr, cl_mem bptr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            bptr,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            nullptr,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned convolution3(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, int iw, int ih, int id, int k, int pad, int ow, int oh, int od)
    {
        cl_int status;
        //Setting kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 3, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 6, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 7, sizeof(int), &k);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 8, sizeof(int), &pad);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(0),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution3_ics(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, int iw, int ih, int id, int k, int pad, int ow, int oh, int od)
    {
        cl_int status;
        //Setting kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(10), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 1, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 2, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 3, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 6, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 7, sizeof(int), &k);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 8, sizeof(int), &pad);

        size_t global_work_size[3];
        global_work_size[0] = iw;
        global_work_size[1] = ih;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(10),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution3_abft(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icsbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                               int iw, int ih, int id, int k, int pad, int ow, int oh, int od, int cscInd)
    {
        //ics
        convolution3(ibuf, wsbuffer, bsbuf, icsbuffer, iw, ih, id, k, pad, ow, oh, 1);

        //convolution layer
        convolution3(ibuf, wbuf, bbuf, obuf, iw, ih, id, k, pad, ow, oh, od);

        //ocs
        output_sum(obuf, ocsbuffer, od, ow, oh);

        //csc
        cs_compare(icsbuffer, ocsbuffer, cscBuf, ow, oh, 1, cscInd);

        return 1;
    }

    unsigned convolution3_abft_icr(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icbuffer, cl_mem isrbuffer, cl_mem isbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                                   int iw, int ih, int id, int k, int pad, int ow, int oh, int od, int cscInd)
    {
        //ic
        convolution3_ics(ibuf, wsbuffer, bsbuf, cicBuf, iw, ih, id, k, pad, ow, oh, od);

        //isr
        output_reduce(cicBuf, cisrBuf, 16, ow, oh, (id / 16));

        //is
        output_sum(cisrBuf, cisBuf, (id / 16), ow, oh);

        //output_sum(cicBuf, cisBuf, id, iw, ih);

        //convolution layer
        convolution3(ibuf, wbuf, bbuf, obuf, iw, ih, id, k, pad, ow, oh, od);

        //ocsr
        output_reduce(obuf, ocsrBuf, 16, ow, oh, (od / 16));

        //ocs
        output_sum(ocsrBuf, ocsbuffer, (od / 16), ow, oh);
        //output_sum(obuf, ocsbuffer, od, ow, oh);

        //csc
        cs_compare(cisBuf, ocsbuffer, cscBuf, ow, oh, 1, cscInd);

        return 1;
    }

    unsigned relu(cl_mem ibuf, cl_mem obuf, int ow, int oh, int od)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *) &obuf);

        size_t global_work_size[3]; //and here
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(4),
                                        3, //was this the issue? ugh
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned relu_d(cl_mem ibuf, cl_mem obuf, int ow, int oh, int od)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(7), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 1, sizeof(cl_mem), (void *) &obuf);

        size_t global_work_size[3]; //and here
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(7),
                                        3, //was this the issue? ugh
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned relu_dmr(cl_mem ibuf, cl_mem dbuf, cl_mem obuf, int ow, int oh, int od, int cscInd)
    {
        relu(ibuf, obuf, ow, oh, od);
        relu_d(ibuf, dbuf, ow, oh, od);

        cs_compare(obuf, dbuf, cscBuf, ow, oh, od, cscInd);

        return 1;
    }

    unsigned maxpool(cl_mem ibuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 2, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 3, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 4, sizeof(int), &kernel_size);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 5, sizeof(int), &stride);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(5),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned maxpool_d(cl_mem ibuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(8), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 2, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 3, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 4, sizeof(int), &kernel_size);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 5, sizeof(int), &stride);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(8),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned maxpool_dmr(cl_mem ibuf, cl_mem dbuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh, int cscInd)
    {
        maxpool(ibuf, obuf, iw, ih, id, stride, kernel_size, ow, oh);
        maxpool_d(ibuf, dbuf, iw, ih, id, stride, kernel_size, ow, oh);

        cs_compare(obuf, dbuf, cscBuf, ow, oh, id, cscInd);

        return 1;
    }

    unsigned output_sum(cl_mem ibuf, cl_mem obuf, int id, int ow, int oh)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *) &obuf);
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
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned output_reduce(cl_mem ibuf, cl_mem obuf, int id, int ow, int oh, int od)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(11), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 2, sizeof(int), &id);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(11),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned cs_compare(cl_mem ibuf, cl_mem obuf, cl_mem csbuf, int ow, int oh, int od, int csInd)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *) &csbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 3, sizeof(int), &csInd);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(3),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[6] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf,
                     int iw, int ow)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 3, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 4, sizeof(int), &iw);

        size_t global_work_size[1];
        global_work_size[0] = ow;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(6),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat_ics(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf,
                         int iw, int ow)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(9), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 2, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 3, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 4, sizeof(int), &iw);

        size_t global_work_size[1];
        global_work_size[0] = ow;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(9),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat_abft(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icbuffer, cl_mem isbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                          int iw, int ow, int cscInd)
    {
        //ic
        flatmat_ics(ibuf, wsbuffer, bsbuf, icbuffer, 32, (iw / 32));

        //is
        output_sum(icbuffer, isbuffer, (iw / 32), 1, 1);

        //matmul
        flatmat(ibuf, wbuf, bbuf, obuf, iw, ow);

        //ocs
        output_sum(obuf, ocsbuffer, ow, 1, 1);

        //csc
        cs_compare(isbuffer, ocsbuffer, cscBuf, 1, 1, 1, cscInd);

        return 1;
    }


    unsigned write_weightsums(double* w01sptr, double* w23sptr, double* w45sptr, double* w56sptr) {
        w01sumBuffer = clCreateBuffer(_ocl_base->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     layer0d * w01w * w01h * sizeof(double),
                                     w01sptr,
                                     NULL);

        w23sumBuffer = clCreateBuffer(_ocl_base->context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      layer3d * w23w * w23h * sizeof(double),
                                      w23sptr,
                                      NULL);

        w45sumBuffer = clCreateBuffer(_ocl_base->context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      layer5d * w45w * w45h * sizeof(double),
                                      w45sptr,
                                      NULL);

        w56sumBuffer = clCreateBuffer(_ocl_base->context,
                                      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      layer5d * layer5h * layer5w * sizeof(double),
                                      w56sptr,
                                      NULL);
    }

    unsigned write_image(double* l0ptr) {
        l0Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer0d * layer0h * layer0w * sizeof(double),
                                  l0ptr,
                                  NULL);
    }

    unsigned write_layerdoubles(double* l1ptr, double* l2ptr, double* l3ptr, double* l4ptr, double* l5ptr) {

        l1rbdBuffer = clCreateBuffer(_ocl_base->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     layer1d * layer1h * layer1w * sizeof(double),
                                     l1ptr,
                                     NULL);

        l2dBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer2d * layer2h * layer2w * sizeof(double),
                                   l2ptr,
                                   NULL);

        l3rbdBuffer = clCreateBuffer(_ocl_base->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     layer3d * layer3h * layer3w * sizeof(double),
                                     l3ptr,
                                     NULL);

        l4dBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer4d * layer4h * layer4w * sizeof(double),
                                   l4ptr,
                                   NULL);

        l5rbdBuffer = clCreateBuffer(_ocl_base->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     layer5d * layer5h * layer5w * sizeof(double),
                                     l5ptr,
                                     NULL);

    }

    unsigned write_layersums(double* l1iptr, double* l1optr, double* l3iptr, double* l3optr, double* l5iptr, double* l5optr, double* l6iptr, double* csptr) {
        l1insumBuffer = clCreateBuffer(_ocl_base->context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     layer1d * layer1h * layer1w * sizeof(double),
                                     l1iptr,
                                     NULL);

        l1outsumBuffer = clCreateBuffer(_ocl_base->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       layer1d * layer1h * layer1w * sizeof(double),
                                       l1optr,
                                       NULL);

        l3insumBuffer = clCreateBuffer(_ocl_base->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       layer3d * layer3h * layer3w * sizeof(double),
                                       l3iptr,
                                       NULL);

        l3outsumBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        layer3d * layer3h * layer3w * sizeof(double),
                                        l3optr,
                                        NULL);

        l5insumBuffer = clCreateBuffer(_ocl_base->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       layer5d * layer5h * layer5w * sizeof(double),
                                       l5iptr,
                                       NULL);

        l5outsumBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        layer5d * layer5h * layer5w * sizeof(double),
                                        l5optr,
                                        NULL);

        l6insumBuffer = clCreateBuffer(_ocl_base->context,
                                       CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(double),
                                       l6iptr,
                                       NULL);

        cscBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   10 * sizeof(double),
                                   csptr,
                                   NULL);
    }

    unsigned write_weights(double* w01ptr, double* w23ptr, double* w45ptr, double* w56ptr)
    {
        w01Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer1d * layer2d * w01w * w01h * sizeof(double),
                                   w01ptr,
                                   NULL);

        w23Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer2d * layer3d * w23w * w23h * sizeof(double),
                                   w23ptr,
                                   NULL);

        w45Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer4d * layer5d * w45w * w45h * sizeof(double),
                                   w45ptr,
                                   NULL);

        w56Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer5d * layer5w * layer5h * OUTPUT * sizeof(double),
                                   w56ptr,
                                   NULL);


    }

    unsigned write_bias(double* b01ptr, double* b23ptr, double* b45ptr, double* b56ptr)
    {
        b01Buffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    layer1d * sizeof(double),
                                    b01ptr,
                                    NULL);

        b23Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer3d * sizeof(double),
                                   b23ptr,
                                   NULL);

        b45Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer5d * sizeof(double),
                                   b45ptr,
                                   NULL);

        b56Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   OUTPUT * sizeof(double),
                                   b56ptr,
                                   NULL);



    }

    unsigned write_layer(double* l1ptr, double* l2ptr, double* l3ptr, double* l4ptr, double* l5ptr, double* l6ptr)
    {
        l1Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer1d * layer1h * layer1w * sizeof(double),
                                   l1ptr,
                                   NULL);

        l1rbBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer1d * layer1h * layer1w * sizeof(double),
                                  l1ptr,
                                  NULL);

        l2Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer2d * layer2h * layer2w * sizeof(double),
                                  l2ptr,
                                  NULL);

        l3Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer3d * layer3h * layer3w * sizeof(double),
                                  l3ptr,
                                  NULL);

        l3rbBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer3d * layer3h * layer3w * sizeof(double),
                                  l3ptr,
                                  NULL);

        l4Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer4d * layer4h * layer4w * sizeof(double),
                                  l4ptr,
                                  NULL);

        l5Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer5d * layer5h * layer5w * sizeof(double),
                                  l5ptr,
                                  NULL);

        l5rbBuffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer5d * layer5h * layer5w * sizeof(double),
                                  l5ptr,
                                  NULL);

        l6Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer6w * sizeof(double),
                                  l6ptr,
                                  NULL);

    }

    unsigned create_bufs_abft()
    {

        icsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                28 * 28 * sizeof(double),
                                nullptr,
                                NULL);

        ficBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                4096 * sizeof(double),
                                nullptr,
                                NULL);

        fisBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                10 * sizeof(double),
                                nullptr,
                                NULL);

        cicBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28 * 512) * sizeof(double),
                                nullptr,
                                NULL);

        cisrBuf = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_WRITE,
                                 (28 * 28 * 64) * sizeof(double), // todo check dimensions
                                 nullptr,
                                 NULL);

        cisBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28) * sizeof(double),
                                nullptr,
                                NULL);


        ocsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                28 * 28 * sizeof(double),
                                nullptr,
                                NULL);

        ocsrBuf = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_WRITE,
                                 (28 * 28 * 64) * sizeof(double),
                                 nullptr,
                                 NULL);

        /*cscBuf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  37 * sizeof(double),
                                  csc,
                                  NULL);*/

        cscBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                37 * sizeof(double),
                                csc,
                                NULL);
    }

    unsigned zero_CSC()
    {
        cscBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                37 * sizeof(double),
                                csc,
                                NULL);
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

    unsigned free_bufs()
    {
        clReleaseMemObject(l0Buffer);

        clReleaseMemObject(icsBuf);
        clReleaseMemObject(ficBuf);
        clReleaseMemObject(fisBuf);
        clReleaseMemObject(cicBuf);
        clReleaseMemObject(cisrBuf);
        clReleaseMemObject(cisBuf);
        clReleaseMemObject(ocsBuf);
        clReleaseMemObject(cscBuf);
    }

    std::unique_ptr<OCL_Base> _ocl_base;

    cl_mem iBuffer = nullptr;
    cl_mem wBuffer = nullptr;
    cl_mem oBuffer = nullptr;
    cl_mem ocsBuffer = nullptr;
    cl_mem icsBuffer = nullptr;
    cl_mem cscBuffer = nullptr;
    cl_mem biasBuffer = nullptr;

    cl_mem icsBuf = nullptr;
    cl_mem ficBuf = nullptr;
    cl_mem fisBuf = nullptr;
    cl_mem cicBuf = nullptr;
    cl_mem cisrBuf = nullptr;
    cl_mem cisBuf = nullptr;
    cl_mem ocsrBuf = nullptr;
    cl_mem ocsBuf = nullptr;
    cl_mem cscBuf = nullptr;
    cl_mem csczBuf = nullptr;

    cl_mem l0Buffer = nullptr;
    cl_mem l1Buffer = nullptr;
    cl_mem l1rbBuffer = nullptr;
    cl_mem l2Buffer = nullptr;
    cl_mem l3Buffer = nullptr;
    cl_mem l3rbBuffer = nullptr;
    cl_mem l4Buffer = nullptr;
    cl_mem l5Buffer = nullptr;
    cl_mem l5rbBuffer = nullptr;
    cl_mem l6Buffer = nullptr;

    cl_mem b01Buffer = nullptr;
    cl_mem b23Buffer = nullptr;
    cl_mem b45Buffer = nullptr;
    cl_mem b56Buffer = nullptr;

    cl_mem w01Buffer = nullptr;
    cl_mem w23Buffer = nullptr;
    cl_mem w45Buffer = nullptr;
    cl_mem w56Buffer = nullptr;

    cl_mem l1insumBuffer = nullptr;
    cl_mem l1outsumBuffer = nullptr;
    cl_mem l1rbdBuffer = nullptr;
    cl_mem l2dBuffer = nullptr;
    cl_mem l3insumBuffer = nullptr;
    cl_mem l3outsumBuffer = nullptr;
    cl_mem l3rbdBuffer = nullptr;
    cl_mem l4dBuffer = nullptr;
    cl_mem l5insumBuffer = nullptr;
    cl_mem l5outsumBuffer = nullptr;
    cl_mem l5rbdBuffer = nullptr;
    cl_mem l6insumBuffer = nullptr;

    cl_mem w01sumBuffer = nullptr;
    cl_mem w23sumBuffer = nullptr;
    cl_mem w45sumBuffer = nullptr;
    cl_mem w56sumBuffer = nullptr;

private:
    cl_program prog_cv;
    cl_program prog_util;

    cl_event _event;

    // 0 - convolution
    // 1 - convolution read
    // 2 - input_sum
    // 3 - input_sum_read
    // 4 - ocs
    // 5 - ocs read
    // 6 - csc
    // 7 - csc
    // 8 - relu
    // 9 - relu read
    unsigned long kernel_execution_times[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

OCL ocl;

static void reset_csc() {
    for (int i=0; i < 37; i++) {
        csc[i] = 0;
    }
    ocl.zero_CSC();
}

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
    matrixW56double = (double*)malloc((layer5d * layer5w * layer5h) * (OUTPUT) * sizeof(double));

    matrixL1insum = (double*)malloc((layer1w * layer1h) * sizeof(double));
    matrixL1outsum = (double*)malloc((layer1w * layer1h) * sizeof(double));
    matrixW01sum = (double*)malloc((layer0d) * (w01w * w01h) * sizeof(double));

    matrixL3insum = (double*)malloc((layer3w * layer3h) * sizeof(double));
    matrixL3outsum = (double*)malloc((layer3w * layer3h) * sizeof(double));
    matrixW23sum = (double*)malloc((layer2d) * (w23w * w23h) * sizeof(double));

    matrixL5insum = (double*)malloc((layer5w * layer5h) * sizeof(double));
    matrixL5outsum = (double*)malloc((layer5w * layer5h) * sizeof(double));
    matrixW45sum = (double*)malloc((layer4d) * (w45w * w45h) * sizeof(double));

    matrixL6insum = (double*)malloc(sizeof(double));
    matrixL6outsum = (double*)malloc(sizeof(double));
    matrixW56sum = (double*)malloc((layer5d * layer5w * layer5h) * sizeof(double));

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
    for (int i = 0; i < (layer1d * layer1h * layer1w); i++) {
        matrixL1double[i] = 0;
    }
    for (int i = 0; i < (layer3d * layer3h * layer3w); i++) {
        matrixL3double[i] = 0;
    }
    for (int i = 0; i < (layer5d * layer5h * layer5w); i++) {
        matrixL5double[i] = 0;
    }
    for (int i = 0; i < (layer1h * layer1w); i++) {
        matrixL1insum[i] = 0;
    }
    for (int i = 0; i < (layer3h * layer3w); i++) {
        matrixL3insum[i] = 0;
    }
    for (int i = 0; i < (layer5h * layer5w); i++) {
        matrixL5insum[i] = 0;
    }
    for (int i = 0; i < (layer1h * layer1w); i++) {
        matrixL1outsum[i] = 0;
    }
    for (int i = 0; i < (layer3h * layer3w); i++) {
        matrixL3outsum[i] = 0;
    }
    for (int i = 0; i < (layer5h * layer5w); i++) {
        matrixL5outsum[i] = 0;
    }
    matrixL6insum[0] = 2;
    matrixL6outsum[0] = 0;

    ics = (double*)malloc(sizeof(double));
    ocs = (double*)malloc(sizeof(double));
    csc = (double*)malloc(10 * sizeof(double));
    for (int i = 0; i < (10); i++) {
        csc[i] = 0;
    }
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

    //matrixW56double
    for (int x0 = 0; x0 < (layer5d * layer5h * layer5w); ++x0)
        for (int x1 = 0; x1 < OUTPUT; ++x1)
            matrixW56double[(x0 * OUTPUT) + (x1)] = lenet->weight5_6[x0][x1];

    //matrixW56sum
    for (int x0 = 0; x0 < (layer5d * layer5h * layer5w); ++x0) {
        matrixW56sum[x0] = 0;
        for (int x1 = 0; x1 < OUTPUT; ++x1) {
            matrixW56sum[x0] += matrixW56double[(x0 * OUTPUT) + (x1)];
        }
    }
    //matrixB01double
    for (int x0 = 0; x0 < layer1d; ++x0) {
        matrixB01double[x0] = lenet->bias0_1[x0];
    }

    //matrixB23double
    for (int x0 = 0; x0 < layer3d; ++x0) {
        matrixB23double[x0] = lenet->bias2_3[x0];
    }

    //matrixB45double
    for (int x0 = 0; x0 < layer5d; ++x0) {
        matrixB45double[x0] = lenet->bias4_5[x0];
    }

    //matrixB56double
    for (int x0 = 0; x0 < OUTPUT; ++x0) {
        matrixB56double[x0] = lenet->bias5_6[x0];
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

static void zero_vectors()
{
    for (int i = 0; i < (layer0h * layer0w * layer0d); i++) {
        matrixL0double[i] = 0;
    }
    for (int i = 0; i < (layer1d * layer1h * layer1w); i++) {
        matrixL1double[i] = 0;
    }
    for (int i = 0; i < (layer2d * layer2h * layer2w); i++) {
        matrixL2double[i] = 0;
    }
    for (int i = 0; i < (layer3d * layer3h * layer3w); i++) {
        matrixL3double[i] = 0;
    }
    for (int i = 0; i < (layer4d * layer4h * layer4w); i++) {
        matrixL4double[i] = 0;
    }
    for (int i = 0; i < (layer5d * layer5h * layer5w); i++) {
        matrixL5double[i] = 0;
    }
    for (int i = 0; i < (OUTPUT); i++) {
        matrixL6double[i] = 0;
    }
    for (int i = 0; i < (layer1h * layer1w); i++) {
        matrixL1insum[i] = 0;
    }
    for (int i = 0; i < (layer1h * layer1w); i++) {
        matrixL1outsum[i] = 0;
    }
    for (int i = 0; i < (layer3h * layer3w); i++) {
        matrixL3insum[i] = 0;
    }
    for (int i = 0; i < (layer3h * layer3w); i++) {
        matrixL3outsum[i] = 0;
    }
    for (int i = 0; i < (layer5h * layer5w); i++) {
        matrixL5insum[i] = 0;
    }
    for (int i = 0; i < (layer5h * layer5w); i++) {
        matrixL5outsum[i] = 0;
    }
    matrixL6insum[0] = 2;
    matrixL6outsum[0] = 0;
    
    for (int i = 0; i < (10); i++) {
        csc[i] = 0;
    }

}

static void forward_ocl(int abft)
{
    int abftflag = 0;
    int counter = 0;

    if (abft == 1) {
        ocl_phase2.setbufs_l01ics();
        ocl_phase2.convolution_nb(layer0w, layer0h, 1, w01w, w01h, layer1w, layer1h, 1, 0, 0);
        counter++;
        //printf("conv layer 1 ics convolutions: %d \n", counter);
    }

    //layer 1 convolution ocl
    ocl_phase2.setbufs_l01();
    //convolution
    counter = 0;
    for (int x = 0; x < (layer0d); ++x) {
        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.convolution_nb(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, x, y);
            counter++;
        }
    }
    //printf("conv layer 0-1 convolutions: %d \n", counter);

    if (abft == 1) {
        ocl_phase2.setbufs_l01ocs();
        ocl_phase2.output_sum_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, 1, 0, 0);

        //for (int i = 0; i < 1; i++) { csc[i] = 0; }
        ocl_phase2.setbufs_l01csc();
        ocl_phase2.cs_compare_nb(0, layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

    }

    //ocs
    ocs[0] = 0;

    //layer1 relu
    ocl_phase2.setbuf_l1rb();

    for (int y = 0; y < layer1d; ++y) {
        ocl_phase2.relu_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, y);
    }

    if (abft == 1) {
        //layer 1 relu + bias doubling and compare
        ocl_phase2.setbuf_l1rbd();

        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.relu_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, y);
        }

        ocl_phase2.setbufs_l1rbcsc();
        ocl_phase2.cs_compare_nb(1, layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
    }


    //layer 2 ocl max pooling
    ocl_phase2.setbuf_l12();

    for (int y = 0; y < layer2d; ++y) {
        ocl_phase2.maxpool_nb(layer1w, layer1h, layer1d, 2, 2, layer2w, layer2h, layer2d, 0, y);
    }

    if (abft == 1) {
        //layer 2 doubling and compare
        ocl_phase2.setbuf_l12d();

        for (int y = 0; y < layer2d; ++y) {
            ocl_phase2.maxpool_nb(layer1w, layer1h, layer1d, 2, 2, layer2w, layer2h, layer2d, 0, y);
        }

        ocl_phase2.setbufs_l2csc();
        ocl_phase2.cs_compare_nb(2, layer2w, layer2h, layer2d, w01w, w01h, layer2w, layer2h, layer2d, 0, 0);
    }

    if (abft == 1) {
        //layer 3 matrix ics:
        counter=0;
        ocl_phase2.setbufs_l23ics();
        for (int x = 0; x < (layer2d); ++x) {
            ocl_phase2.convolution_nb(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, x, 0);
            counter++;
        }
    }

    //layer3 convolution
    ocl_phase2.setbufs_l23();

    counter=0;
    for (int x = 0; x < (layer2d); ++x) {
        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.convolution_nb(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, x, y);
            counter++;
        }
    }
    //printf("conv layer 2-3 convolutions: %d \n",counter);

    if (abft == 1) {
        ocl_phase2.setbufs_l23ocs();
        ocl_phase2.output_sum_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, 1, 0, 0);

        ocl_phase2.setbufs_l23csc();
        ocl_phase2.cs_compare_nb(3, layer3w, layer3h, 1, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    }

    ocl_phase2.setbuf_l3rb();

    //layer3 relu
    for (int y = 0; y < layer3d; ++y) {
        ocl_phase2.relu_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, y);
    }

    if (abft == 1) {
        //layer 3 relu + bias doubling and compare
        ocl_phase2.setbuf_l3rbd();

        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.relu_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, y);
        }

        ocl_phase2.setbufs_l3rbcsc();
        ocl_phase2.cs_compare_nb(4, layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    }

    //layer4 ocl max pooling
    ocl_phase2.setbuf_l34();
    for (int y = 0; y < layer4d; ++y) {
        ocl_phase2.maxpool_nb(layer3w, layer3h, layer3d, 2, 2, layer4w, layer4h, layer4d, 0, y);
    }

    if (abft == 1) {
        //layer 4 max pool doubling and compare
        ocl_phase2.setbuf_l4d();

        for (int y = 0; y < layer4d; ++y) {
            ocl_phase2.maxpool_nb(layer3w, layer3h, layer3d, 2, 2, layer4w, layer4h, layer4d, 0, y);
        }

        ocl_phase2.setbufs_l2csc();
        ocl_phase2.cs_compare_nb(5, layer4w, layer4h, layer4d, w01w, w01h, layer4w, layer4h, layer4d, 0, 0);
    }

    if (abft == 1) {
        //layer 5 matrix ics:
        ocl_phase2.setbufs_l45ics();
        counter =0;
        for (int x = 0; x < (layer4d); ++x) {
            ocl_phase2.convolution_nb(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, 1, x, 0);
            counter++;
        }
        //printf("conv layer 5 ics convolutions: %d \n",counter);
    }


    ocl_phase2.setbufs_l45();

    counter = 0;
    for (int x = 0; x < (layer4d); ++x) {
        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.convolution_nb(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, x, y);
            counter++;
        }
    }
    //printf("conv layer 4-5 convolutions: %d \n",counter);

    if (abft == 1) {
        //layer 5 output cs
        ocl_phase2.setbufs_l45ocs();
        ocl_phase2.output_sum_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, 1, 0, 0);

        ocl_phase2.setbufs_l45csc();
        ocl_phase2.cs_compare_nb(6, layer5w, layer5h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0);
        //there is a problem with reading and writing beyond index 4/5? to the csc or ocs csc buf
    }


    //layer5 relu
    ocl_phase2.setbuf_l5rb();
    for (int y = 0; y < layer5d; ++y) {
        ocl_phase2.relu_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, y);
    }

    if (abft == 1) {
        //layer 5 relu + bias doubling and compare
        ocl_phase2.setbuf_l5rbd();

        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.relu_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, y);
        }

        ocl_phase2.setbufs_l5rbcsc();
        ocl_phase2.cs_compare_nb(7, layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0);
    }

    //ocl_phase2.last_read(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0, matrixL5double);

    /*for (int i = 0; i < layer5d; ++i) {
        for (int j = 0; j < layer5h; ++j) {
            for (int k = 0; k < layer5w; ++k) {
                printf("%f ", matrixL5double[(i * layer5h * layer5w) + (j * layer5w) + k]);
            }
            printf("\n");
        }
    }*/

    //abft csc results:
    if (abft == 1) {
        ocl_phase2.cs_compare_read(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, csc);

        for (int i = 0; i < 10; i++) {
            if (csc[i] != 0) {
                abftflag = 1;
            }
        }

        if (abftflag == 1) {
            printf("csc: \n");
            for (int i = 0; i < 10; i++) {
                printf("%f ", csc[i]);
            }
            printf("\n");
        }
    }

    //output layer matrix multiplication in c++
    /*for (int y = 0; y < (OUTPUT); ++y) {
        matrixL6double[y] = 0;
    }

    //printf("L5 and W56: \n");

    for (int x = 0; x < (layer5d * layer5h * layer5w); ++x) {
        for (int y = 0; y < (OUTPUT); ++y) {
            matrixL6double[y] += matrixL5double[x] * matrixW56double[x * (OUTPUT) + y];
            //printf("W56: %f \n",matrixW56double[x * (OUTPUT) + y]);
        }
        //printf("L5: %f \n",matrixL5double[x]);
    }
    //printf("\n");*/

    if (abft == 1) {
        //flattened matrix ics
        ocl_phase2.setbuf_l56ics();
        ocl_phase2.flatmat_ics(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0);
        ocl_phase2.flatmat_ics_read(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0,
                                matrixL6insum);
    }
    //printf("l6insum: \n");
    //printf(" %f", matrixL6insum[0]);
    //printf("\n");

    //flattened matrix multiplication
    ocl_phase2.setbuf_l56();
    ocl_phase2.flatmat_nb(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0);

    ocl_phase2.last_read(layer6w, layer6h, layer6d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0, matrixL6double);

    ocs[0] = 0;

    if (abft == 1) {

        //printf("l6 ics: %f\n", ics[0]);

        //printf("L6: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ",matrixL6double[i]);
            ocs[0] += matrixL6double[i];
        }
        //printf("\n");

        //printf("l6 ics with bias: %f\n", ics[0]);
        //printf("l6 ocs: %f\n", ocs[0]);

        if (abs(matrixL6insum[0] - ocs[0]) > 0.00001) {
            printf("l6 ics: %f\n", matrixL6insum[0]);
            printf("l6 ocs: %f\n", ocs[0]);
            abftflag = 1;
        }
    }

    for (int j = 0; j < (OUTPUT); ++j) {
        if (matrixL6double[j] < 0) {
            matrixL6double[j] = 0;
        }
    }

    /*for (int j = 0; j < (OUTPUT); ++j) {
        if (matrixL6double[j] + matrixB56double[j] > 0) {
            matrixL6double[j] += matrixB56double[j];
        } else {
            matrixL6double[j] = 0;
        }
    }*/

    /*printf("L6: ");
    for (uint8 i = 1; i < OUTPUT; ++i) {
        printf("%f ",matrixL6double[i]);
    }
    printf("\n");*/
    //printf("abft flag: %d\n", abftflag);
    abft_err = abftflag;

}

static void forward() {
    //layer 0 -> 1
    ocl.convolution3(ocl.l0Buffer, ocl.w01Buffer, ocl.b01Buffer, ocl.l1Buffer,
                     layer0w, layer0h, layer0d, w01w, 0, layer1w, layer1h, layer1d);

    //check_layer();
    ocl.relu(ocl.l1Buffer, ocl.l1rbBuffer, layer1w, layer1h, layer1d);
    //check_layer();

    //layer 1 -> 2
    ocl.maxpool(ocl.l1rbBuffer, ocl.l2Buffer, layer1w, layer1h, layer1d, 2, 2, layer2w, layer2h);

    //layer 2 -> 3
    ocl.convolution3(ocl.l2Buffer, ocl.w23Buffer, ocl.b23Buffer, ocl.l3Buffer,
                     layer2w, layer2h, layer2d, w23w, 0, layer3w, layer3h, layer3d);

    ocl.relu(ocl.l3Buffer, ocl.l3rbBuffer, layer3w, layer3h, layer3d);

    //layer 3 -> 4
    ocl.maxpool(ocl.c11Buf, ocl.c21Buf, c1w, c1h, c1d, 2, 2, c2w, c2h);


    //layer 4 -> 5
    ocl.convolution3(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                     c2w, c2h, c20d, k2, c2pad, c2w, c2h, c2d);
    ocl.relu(ocl.c22Buf, ocl.c21Buf, c2w, c2h, c2d);


    //layer 5 -> 6
    ocl.flatmat(ocl.c63Buf, ocl.w63Buffer, ocl.b63Buffer, ocl.c62Buf,
                4096, 1000);
    ocl.relu(ocl.c62Buf, ocl.c6rBuf, 1000, 1, 1);
}

int forward_abft() {
    int abftflag = 0;

    //layer 0 -> 1
    ocl.convolution3_abft(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c12Buf,
                          ocl.icsBuf, ocl.w11sBuffer, ocl.b11sBuffer, ocl.ocsBuf,
                          c1w, c1h, c10d, k1, c1pad, c1w, c1h, c1d, 0);

    ocl.relu_dmr(ocl.c12Buf, ocl.c1dBuf, ocl.c11Buf, c1w, c1h, c1d, 1);

    //layer 1 -> 2
    ocl.maxpool_dmr(ocl.c11Buf, ocl.c2dBuf, ocl.c21Buf, c1w, c1h, c1d, 2, 2, c2w, c2h, 4);


    //layer 2 -> 3
    ocl.convolution3_abft(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                          ocl.icsBuf, ocl.w21sBuffer, ocl.b21sBuffer, ocl.ocsBuf,
                          c2w, c2h, c20d, k2, c2pad, c2w, c2h, c2d, 5);
    ocl.relu_dmr(ocl.c22Buf, ocl.c2dBuf, ocl.c21Buf, c2w, c2h, c2d, 6);

    //layer 3 -> 4
    ocl.maxpool_dmr(ocl.c21Buf, ocl.c3dBuf, ocl.c31Buf, c2w, c2h, c2d, 2, 2, c3w, c3h, 9);


    //layer 4 -> 5
    ocl.convolution3_abft(ocl.c31Buf, ocl.w31Buffer, ocl.b31Buffer, ocl.c32Buf,
                          ocl.icsBuf, ocl.w31sBuffer, ocl.b31sBuffer, ocl.ocsBuf,
                          c3w, c3h, c30d, k3, c3pad, c3w, c3h, c3d, 10);
    ocl.relu_dmr(ocl.c32Buf, ocl.c3dBuf, ocl.c31Buf, c3w, c3h, c3d, 11);

    //layer 5 -> 6
    ocl.flatmat_abft(ocl.c63Buf, ocl.w63Buffer, ocl.b63Buffer, ocl.c62Buf,
                     ocl.ficBuf, ocl.fisBuf, ocl.w63sBuffer, ocl.b63sBuffer, ocl. ocsBuf,
                     4096, 1000,  35);
    ocl.relu_dmr(ocl.c62Buf, ocl.c6dBuf, ocl.c6rBuf, 1000, 1, 1, 36);

    return abftflag;
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

uint8 Predict(LeNet5 *lenet, image input, uint8 count)
{
    Feature features = { 0 };
    load_input(&features, input);
    forward(lenet, &features, relu);

    //print L6 floats here
    //printf("layer6 c++: ");
    for (uint8 i = 0; i < OUTPUT; ++i) {
        //printf("%f ", features.output[i]);
        if (abs(matrixL6double[i] - features.output[i]) > 0.0000000000001) {
            printf("output mismatch: ocl: %f c++: %f \n", matrixL6double[i], features.output[i]);
        }
    }
    //printf("\n");

    //printf("c++ result: %d \n", get_result(&features, count));

    return get_result(&features, count);
}

uint8 Predict_ocl(image input, int abft, uint8 count)
{

    load_input_ocl(input);
    ocl.write_image(matrixL0double);

    forward_ocl(abft);

    //print L6 floats here
    //printf("layer6 Ocl: ");
    for (uint8 i = 0; i < OUTPUT; ++i) {
        //printf("%f ", matrixL6double[i]);
    }
    //printf("\n");

    //getting result from the output matrix/vector
    const int outlen = OUTPUT;
    uint8 result = 0;
    double maxvalue = 0;
    for (uint8 i = 0; i < OUTPUT; ++i) {
        if (matrixL6double[i] > maxvalue) {
            maxvalue = matrixL6double[i];
            result = i;
        }
    }

    if (abft_err != 0) {
        printf("abft flag raised\n");
        abft_err = 0;
    }

    //printf("ocl prediction: %d\n",result);
    return result;
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

int init_mem(int abft) {
    zero_vectors();
    ocl.write_layer(matrixL1double, matrixL2double, matrixL3double, matrixL4double, matrixL5double, matrixL6double);

    if (abft == 1) {
        ocl.write_layersums(matrixL1insum, matrixL1outsum, matrixL3insum, matrixL3outsum, matrixL5insum, matrixL5outsum, matrixL6insum, csc);
        ocl.write_layerdoubles(matrixL1double, matrixL2double, matrixL3double, matrixL4double, matrixL5double);
    }

}


int testing(LeNet5 *lenet, image *test_data, uint8 *test_label,int total_size)
{
    int right = 0, percent = 0;
    for (int i = 0; i < total_size; ++i)
    {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        //printf("prediction c++: %d \n", p);
        right += l == p;
        /*if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);*/
    }
    return right;
}

int testing_ocl(image *test_data, uint8 *test_label, int abft, int total_size)
{
    int right = 0, percent = 0;
    int acctemp = 0;
    for (int i = 0; i < total_size; ++i)
    {
        init_mem(abft);
        uint8 l = test_label[i];
        int p = Predict_ocl(test_data[i], 1, 10);
        //printf("prediction ocl: %d \n", p);
        right += l == p;
        acctemp += l == p;
        if (i % 100 == 0 && i != 0) {
            printf("ocl accuracy: %d / 100\n", acctemp);
            printf("ocl total accuracy: %d / %d \n", right, i);
            acctemp = 0;
        }
        /*if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);*/
    }
    return right;
}

int testing_comb(LeNet5 *lenet, image *test_data, uint8 *test_label, int abft, int total_size)
{
    int right = 0, percent = 0;
    int acctemp = 0;
    for (int i = 0; i < total_size; ++i)
    {
        init_mem(abft);
        uint8 l = test_label[i];
        int p = Predict_ocl(test_data[i], abft, 10);
        int pp = Predict(lenet, test_data[i], 10);
        //printf("prediction ocl: %d \n", p);

        if (p != pp) {
            printf("Prediction mismatch! ocl:%d, c++:%d \n", p, pp);
        }

        right += l == p;
        acctemp += l == p;
        if (i % 100 == 0 && i != 0) {
            printf("ocl accuracy: %d / 100\n", acctemp);
            printf("ocl total accuracy: %d / %d \n", right, i);
            acctemp = 0;
        }
        /*if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);*/
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

struct Predict_noabft : public IProgram {
    int run() override {

        forward_ocl(0);

        //print L6 floats here
        //printf("layer6 Ocl: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ", matrixL6double[i]);
        }
        //printf("\n");

        //getting result from the output matrix/vector
        const int outlen = OUTPUT;
        uint8 result = 0;
        double maxvalue = 0;
        for (uint8 i = 0; i < OUTPUT; ++i) {
            if (matrixL6double[i] > maxvalue) {
                maxvalue = matrixL6double[i];
                result = i;
            }
        }

        //printf("ocl prediction: %d\n",result);
        return result;
    }
};

struct Predict_abft : public IProgram {
    int run() override {

        forward_ocl(1);

        //print L6 floats here
        //printf("layer6 Ocl: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ", matrixL6double[i]);
        }
        //printf("\n");

        //getting result from the output matrix/vector
        const int outlen = OUTPUT;
        uint8 result = 0;
        double maxvalue = 0;
        for (uint8 i = 0; i < OUTPUT; ++i) {
            if (matrixL6double[i] > maxvalue) {
                maxvalue = matrixL6double[i];
                result = i;
            }
        }

        if (abft_err != 0) {
            printf("abft flag raised\n");
            abft_err = 0;
        }

        //printf("ocl prediction: %d\n",result);
        return result;
    }
};

int main() {
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    //Program
    Predict_noabft predictNoabft;
    Predict_abft predictAbft;

    int result = 0;

    image *test_data = (image *) calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *) calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR!\nDataset File Not Found! Please Copy Dataset to the Folder Including the exe\n");
        free(test_data);
        free(test_label);
        system("pause");
    }


    LeNet5 *lenet = (LeNet5 *) malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);

    createVectors();
    copyModel(lenet);

    int abft_enable = 1;

    ocl.write_weights(matrixW01double, matrixW23double, matrixW45double, matrixW56double);
    ocl.write_bias(matrixB01double, matrixB23double, matrixB45double, matrixB56double);

    if (abft_enable == 1) {
        ocl.write_weightsums(matrixW01sum, matrixW23sum, matrixW45sum, matrixW56sum);
    }
    double time1 = 0;
    double time2 = 0;

    for (int i= 0; i < 0; i++) {
        //ocl_phase2.free_bufs();
        init_mem(0);
        load_input_ocl(test_data[i]);
        ocl.write_image(matrixL0double);
        result = Program_sw.runProgram(predictNoabft);
        time1 += Program_sw.getElapsedTime();
    }
    std::cout << "single prediction without abft: " << result << std::endl;
    std::cout << "Elapsed time: " << time1 << " us" << std::endl;

    for (int i= 0; i < 0; i++) {
        //ocl_phase2.free_bufs();
        init_mem(1);
        load_input_ocl(test_data[i]);
        ocl.write_image(matrixL0double);
        result = Program_sw.runProgram(predictAbft);
        time2 += Program_sw.getElapsedTime();
    }
    std::cout << "single prediction with abft: " << result << std::endl;
    std::cout << "Elapsed time: " << time2 << " us" << std::endl;
    std::cout << "Time difference: " << time2 - time1 << " us" << std::endl;
    std::cout << "abft overhead: " << ((time2 - time1) / time1) << " " << std::endl;
    

    //int right = testing(lenet, test_data, test_label, COUNT_TEST);
    //int right = testing(lenet, test_data, test_label, 100);
    //printf("c++ right: %d / 100 \n", right);

    //int right_ocl = testing_ocl(test_data, test_label, 100);
    //printf("ocl accuracy: %d / %d \n", right_ocl, 100);

    int test_len = 1;

    for (int i = 0; i < test_len; i++) {
        int right_comb = testing_comb(lenet, test_data, test_label, abft_enable,  test_len);
        printf("accuracy: %d / %d \n", right_comb, test_len);
    }
    // p = Predict(lenet, test_data[120], 10);
    //int oclp = Predict_ocl(test_data[120], 10);
    //printf("c: %d, ocl: %d \n",p, oclp);

    //printf("%d/%d\n", right, COUNT_TEST);
    free(lenet);
    free(test_data);
    free(test_label);

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl.print_kernel_execution_times();

    ocl.free_bufs();

    free(matrixL0double);
    free(matrixL1double);
    free(matrixL2double);
    free(matrixL3double);
    free(matrixL4double);
    free(matrixL5double);
    free(matrixL6double);

    free(matrixL1insum);
    free(matrixL1outsum);

    free(matrixL3insum);
    free(matrixL3outsum);

    free(matrixL5insum);
    free(matrixL5outsum);

    free(matrixL6insum);
    free(matrixL6outsum);

    free(matrixW01double);
    free(matrixW12double);
    free(matrixW23double);
    free(matrixW34double);
    free(matrixW45double);
    free(matrixW56double);

    free(matrixW01sum);
    free(matrixW23sum);
    free(matrixW45sum);
    free(matrixW56sum);

    free(matrixB01double);
    free(matrixB12double);
    free(matrixB23double);
    free(matrixB34double);
    free(matrixB45double);
    free(matrixB56double);

    free(ics);
    free(ocs);
    free(csc);

    printPlatformInfo(false);
    return 0;
}
