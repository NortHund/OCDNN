#include "vgg.h"

int layer0w = 224;
int layer0h = 224;
int layer0d = 3;

int c1w = 224;
int c1h = 224;
int c1d = 64;

int c2w = 112;
int c2h = 112;
int c2d = 128;

int c3w = 56;
int c3h = 56;
int c3d = 256;

int k1 = 3;
int k2 = 3;
int k3 = 3;

int c1pad = 1;
int c2pad = 1;
int c3pad = 1;

double* matrixL0double;

double* matrixW11double;
double* matrixW12double;
double* matrixW21double;
double* matrixW22double;

double* matrixB11double;
double* matrixB12double;
double* matrixB21double;
double* matrixB22double;

double* matrixR;
double* matrixR2;

int abft_err = 0;

int freememory() {
    free(matrixL0double);

    free(matrixW11double);
    free(matrixW12double);

    free(matrixB11double);
    free(matrixB12double);

    free(matrixR);
    free(matrixR2);
}

static void createVectors()
{
    matrixL0double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));

    matrixW11double = (double*)malloc((layer0d) * (c1d) * (k1 * k1) * sizeof(double));
    matrixW12double = (double*)malloc((c1d) * (c1d) * (k1 * k1) * sizeof(double));
    matrixW21double = (double*)malloc((c2d) * (c2d) * (k2 * k2) * sizeof(double));
    matrixW22double = (double*)malloc((c2d) * (c2d) * (k2 * k2) * sizeof(double));

    matrixB11double = (double*)malloc((c1d) * sizeof(double));
    matrixB12double = (double*)malloc((c1d) * sizeof(double));
    matrixB21double = (double*)malloc((c2d) * sizeof(double));
    matrixB22double = (double*)malloc((c2d) * sizeof(double));

    matrixR = (double*)malloc((c1d) * (c1w * c1h) * sizeof(double));
    matrixR2 = (double*)malloc((c2d) * (c2w * c2h) * sizeof(double));


    for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL0double[i] = 0;
    }

    for (int i = 0; i < (layer0d * c1d * k1 * k1); i++) {
        matrixW11double[i] = 0.7;
    }
    for (int i = 0; i < (c1d * c1d * k1 * k1); i++) {
        matrixW12double[i] = 0.007;
    }
    for (int i = 0; i < (c2d * c2d * k2 * k2); i++) {
        matrixW21double[i] = 0.007;
        matrixW22double[i] = 0.007;
    }

    for (int i = 0; i < (c1d); i++) {
        matrixB11double[i] = 0.2;
        matrixB12double[i] = 0.2;
    }
    for (int i = 0; i < (c2d); i++) {
        matrixB21double[i] = 0.2;
        matrixB22double[i] = 0.2;
    }

}

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
        create_layers();
    }

    ~OCL()
    {
        free_bufs();
    }

    void init_programs()
    {
        prog_cv = _ocl_base->CreateProgramFromFile("kernels/vgg-conv.cl");
        prog_util = _ocl_base->CreateProgramFromFile("kernels/vgg-util.cl");
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
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat_ics"); //7
    }

    cl_mem l0Buffer = nullptr;

    cl_mem b11Buffer = nullptr;
    cl_mem b12Buffer = nullptr;
    cl_mem b21Buffer = nullptr;
    cl_mem b22Buffer = nullptr;

    cl_mem w11Buffer = nullptr;
    cl_mem w12Buffer = nullptr;
    cl_mem w21Buffer = nullptr;
    cl_mem w22Buffer = nullptr;

    cl_mem c11Buf = nullptr;
    cl_mem c12Buf = nullptr;

    cl_mem c21Buf = nullptr;
    cl_mem c22Buf = nullptr;

    cl_mem c31Buf = nullptr;
    cl_mem c32Buf = nullptr;

    cl_mem c41Buf = nullptr;
    cl_mem c42Buf = nullptr;

    cl_mem c51Buf = nullptr;
    cl_mem c52Buf = nullptr;

    unsigned create_layers()
    {
        c11Buf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  c1d * c1h * c1w * sizeof(double),
                                  nullptr,
                                  NULL);

        c12Buf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  c1d * c1h * c1w * sizeof(double),
                                  nullptr,
                                  NULL);

        c21Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2d * c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        c22Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2d * c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        c31Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c3d * c3h * c3w * sizeof(double),
                                nullptr,
                                NULL);

        c32Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c3d * c3h * c3w * sizeof(double),
                                nullptr,
                                NULL);

    }

    unsigned free_bufs()
    {
        clReleaseMemObject(l0Buffer);

        clReleaseMemObject(b11Buffer);
        clReleaseMemObject(b12Buffer);

        clReleaseMemObject(w11Buffer);
        clReleaseMemObject(w12Buffer);

        clReleaseMemObject(c21Buf);
        clReleaseMemObject(c22Buf);

        clReleaseMemObject(c31Buf);
        clReleaseMemObject(c32Buf);

        clReleaseMemObject(c41Buf);
        clReleaseMemObject(c42Buf);

        clReleaseMemObject(c51Buf);
        clReleaseMemObject(c52Buf);
    }

    unsigned write_image(double* l0ptr) {
        l0Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer0d * layer0h * layer0w * sizeof(double),
                                  l0ptr,
                                  NULL);
    }

    unsigned write_weights(double* w11ptr, double* w12ptr, double* w21ptr, double* w22ptr)
    {
        w11Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer0d * c1d * k1 * k1 * sizeof(double),
                                   w11ptr,
                                   NULL);

        w12Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c1d * c1d * k1 * k1 * sizeof(double),
                                   w12ptr,
                                   NULL);

        w21Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * c2d * k1 * k1 * sizeof(double),
                                   w21ptr,
                                   NULL);

        w22Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * c2d * k1 * k1 * sizeof(double),
                                   w22ptr,
                                   NULL);

    }

    unsigned write_bias(double* b11ptr, double* b12ptr, double* b21ptr, double* b22ptr)
    {
        b11Buffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c1d * sizeof(double),
                                    b11ptr,
                                    NULL);

        b12Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c1d * sizeof(double),
                                   b12ptr,
                                   NULL);

        b21Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * sizeof(double),
                                   b21ptr,
                                   NULL);

        b22Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * sizeof(double),
                                   b22ptr,
                                   NULL);

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

    unsigned convolution3(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, int iw, int ih, int id, int k, int ow, int oh, int od, int pad)
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

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned relu(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(4), 3, sizeof(int), &olm);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(4),
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

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned output_sum(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
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

    unsigned cs_compare(int layer, int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(3), 3, sizeof(int), &layer);

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

    unsigned flatmat(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(6), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 6, sizeof(int), &iw);

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
            std::cerr << "At d: " <<  iln << " d2: " << olm  << std::endl;
            exit(EXIT_FAILURE);
        }

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat_ics(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(7), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 6, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 7, sizeof(int), &ow);

        size_t global_work_size[1];
        global_work_size[0] = 1;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(7),
                                        1,
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

int load_image(const char* filename)
{
    std::vector<unsigned char> L0char;

    unsigned width;
    unsigned height;

    //unsigned output = lodepng_decode32_file(&L0char,&width, &height, filename);
    unsigned output = lodepng::decode(L0char, width, height, filename);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;


    /*for (int i = 0; i < 10; i++) {
        printf("%d ", L0char[i]);
    }
    printf("\n");*/

    for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL0double[i] = L0char[i];
    }

    return 1;
}

static void forward_ocl(int abft)
{
    int abftflag = 0;

    //conv block 1
    //convolution 1-1
    ocl.convolution3(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c11Buf,
                     c1w, c1h, c1d, k1, c1w, c1d, c1d, c1pad);

    //convolution 1-2
    ocl.convolution3(ocl.c11Buf, ocl.w12Buffer, ocl.b12Buffer, ocl.c12Buf,
                     c1w, c1h, c1d, k1, c1w, c1h, c1d, c1pad);

    //max pool 1
    ocl.maxpool(ocl.c12Buf, ocl.c21Buf, c1w, c1h, c1d, 2, 4, c2w, c2h);

    //conv block 2
    //convolution 2-1
    ocl.convolution3(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                     c2w, c2h, c2d, k2, c2w, c2d, c2d, c2pad);

    //convolution 2-2
    ocl.convolution3(ocl.c22Buf, ocl.w22Buffer, ocl.b22Buffer, ocl.c21Buf,
                     c2w, c2h, c2d, k2, c2w, c2h, c2d, c2pad);

    //max pool 2
    ocl.maxpool(ocl.c21Buf, ocl.c31Buf, c2w, c2h, c2d, 2, 4, c3w, c3h);





    //with 0 bias
    /*ocl.convolution3(ocl.l0Buffer, ocl.w01Buffer, nullptr, ocl.l1Buffer,
                     layer0w, layer0h, layer0d, k01, layer1w, layer1h, layer1d);*/

    ocl.buf_read(c2w, c2h, c2d, matrixR2, ocl.c21Buf);

    for (int i=100; i <110 ; i++) {
        printf("%f ", matrixR2[i]);
    }


    abft_err = abftflag;
}



int main() {
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    int result = 0;

    createVectors();

    load_image("../../source-img/in0.png");

    for (int i=0; i <(layer0d * layer0h * layer0w) ; i++) {
        matrixL0double[i] = 1;
    }

    ocl.write_image(matrixL0double);

    ocl.create_layers();
    ocl.write_weights(matrixW11double, matrixW12double, matrixW21double, matrixW22double);
    ocl.write_bias(matrixB11double, matrixB12double, matrixB21double, matrixB22double);

    forward_ocl(1);

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    //cleaning bufs and memory allocation
    freememory();

    //print opencl information
    printPlatformInfo(false);

    return 0;
}
