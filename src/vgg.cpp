#include "vgg.h"

int layer0w = 24;
int layer0h = 24;
int layer0d = 3;

int c1w = 24;
int c1h = 24;
int c1d = 3;

int c2w = 12;
int c2h = 12;
int c2d = 3;

int c3w = 6;
int c3h = 6;
int c3d = 3;

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


double* matrixW11sum;
double* matrixW12sum;
double* matrixW21sum;
double* matrixW22sum;

double* matrixB11sum;
double* matrixB12sum;
double* matrixB21sum;
double* matrixB22sum;

double* ics;
double* ocs;
double* csc;

double* matrixR;
double* matrixR2;

int abft_err = 0;

int freememory() {
    free(matrixL0double);

    free(matrixW11double);
    free(matrixW12double);
    free(matrixW21double);
    free(matrixW22double);

    free(matrixB11double);
    free(matrixB12double);
    free(matrixB21double);
    free(matrixB22double);

    free(matrixR);
    free(matrixR2);

    free(matrixW11sum);
    free(matrixW12sum);
    free(matrixW21sum);
    free(matrixW22sum);

    free(matrixB11sum);
    free(matrixB12sum);
    free(matrixB21sum);
    free(matrixB22sum);

    free(ics);
    free(ocs);
    free(csc);
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

    matrixW11sum = (double*)malloc((c1d) * (k1 * k1) * sizeof(double));
    matrixW12sum = (double*)malloc((c1d) * (k1 * k1) * sizeof(double));
    matrixW21sum = (double*)malloc((c2d) * (k2 * k2) * sizeof(double));
    matrixW22sum = (double*)malloc((c2d) * (k2 * k2) * sizeof(double));

    matrixB11sum = (double*)malloc(sizeof(double));
    matrixB12sum = (double*)malloc(sizeof(double));
    matrixB21sum = (double*)malloc(sizeof(double));
    matrixB22sum = (double*)malloc(sizeof(double));

    ics = (double*)malloc((c2w * c2h) * sizeof(double));
    ocs = (double*)malloc((c2w * c2h) * sizeof(double));
    csc = (double*)malloc((32) * sizeof(double));

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
        matrixB11double[i] = 0.1;
        matrixB12double[i] = 0.2;
    }
    for (int i = 0; i < (c2d); i++) {
        matrixB21double[i] = 0.3;
        matrixB22double[i] = 0.4;
    }

}

static void copyModel() {
    //1-1
    FILE *fp = fopen("../../source-data/vggbin/0.bin", "rb");
    fread(matrixW11double, 1728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vggbin/1.bin", "rb");
    fread(matrixB11double, 64 * sizeof(double), 1, fp);
    fclose(fp);

    //1-2
    fp = fopen("../../source-data/vggbin/2.bin", "rb");
    fread(matrixW12double, 1728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vggbin/3.bin", "rb");
    fread(matrixB12double, 64 * sizeof(double), 1, fp);
    fclose(fp);

    //2-1 -sizes?
    fp = fopen("../../source-data/vggbin/4.bin", "rb");
    fread(matrixW12double, 1728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vggbin/5.bin", "rb");
    fread(matrixB12double, 64 * sizeof(double), 1, fp);
    fclose(fp);

    //2-2 -sizes?
    fp = fopen("../../source-data/vggbin/6.bin", "rb");
    fread(matrixW12double, 1728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vggbin/7.bin", "rb");
    fread(matrixB12double, 64 * sizeof(double), 1, fp);
    fclose(fp);



}

static void create_weight_sums() {
    for (int i = 0; i < (c1d * k1 * k1); i++) {
        matrixW11sum[i] = 0;
        matrixW12sum[i] = 0;
    }
    for (int i = 0; i < (c2d * k2 * k2); i++) {
        matrixW21sum[i] = 0;
        matrixW22sum[i] = 0;
    }

    matrixB11sum[0] = 0;
    matrixB12sum[0] = 0;
    matrixB21sum[0] = 0;
    matrixB22sum[0] = 0;


    for (int h = 0; h < c1d; h++) {
        for (int i = 0; i < c1d; i++) {
            for (int j = 0; j < k1; j++) {
                for (int k = 0; k < k1; k++) {
                    matrixW11sum[(i * k1 * k1) + (j * k1) + k] += matrixW11double[(h * c1d * k1 * k1) + (i * k1 * k1) + (j * k1) + k];
                    matrixW12sum[(i * k1 * k1) + (j * k1) + k] += matrixW12double[(h * c1d * k1 * k1) + (i * k1 * k1) + (j * k1) + k];
                }
            }
        }
    }

    for (int h = 0; h < c2d; h++) {
        for (int i = 0; i < c2d; i++) {
            for (int j = 0; j < k2; j++) {
                for (int k = 0; k < k2; k++) {
                    matrixW21sum[(i * k2 * k2) + (j * k2) + k] += matrixW21double[(h * c2d * k2 * k2) + (i * k2 * k2) + (j * k2) + k];
                    matrixW22sum[(i * k2 * k2) + (j * k2) + k] += matrixW22double[(h * c2d * k2 * k2) + (i * k2 * k2) + (j * k2) + k];
                }
            }
        }
    }

    for (int h = 0; h < c1d; h++) {
        matrixB11sum[0] += matrixB11double[h];
        matrixB12sum[0] += matrixB12double[h];
    }
    for (int h = 0; h < c2d; h++) {
        matrixB21sum[0] += matrixB21double[h];
        matrixB22sum[0] += matrixB22double[h];
    }

    /*for (int i = 0; i < c2d; i++) {
        for (int j = 0; j < k2; j++) {
            for (int k = 0; k < k2; k++) {
                printf("%f ", matrixW22sum[(i * k2 * k2) + (j * k2) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }*/

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
        create_bufs_abft();
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

    cl_mem w21sBuffer = nullptr;
    cl_mem w22sBuffer = nullptr;

    cl_mem b11sBuffer = nullptr;
    cl_mem b12sBuffer = nullptr;
    cl_mem b21sBuffer = nullptr;
    cl_mem b22sBuffer = nullptr;

    cl_mem icsBuf = nullptr;
    cl_mem ocsBuf = nullptr;
    cl_mem cscBuf = nullptr;

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

    unsigned create_bufs_abft()
    {
        icsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        ocsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        cscBuf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  32 * sizeof(double),
                                  nullptr,
                                  NULL);
    }

    unsigned free_bufs()
    {
        clReleaseMemObject(l0Buffer);

        clReleaseMemObject(b11Buffer);
        clReleaseMemObject(b12Buffer);
        clReleaseMemObject(b21Buffer);
        clReleaseMemObject(b22Buffer);

        clReleaseMemObject(w11Buffer);
        clReleaseMemObject(w12Buffer);
        clReleaseMemObject(w21Buffer);
        clReleaseMemObject(w22Buffer);

        clReleaseMemObject(c21Buf);
        clReleaseMemObject(c22Buf);

        clReleaseMemObject(c31Buf);
        clReleaseMemObject(c32Buf);

        clReleaseMemObject(c41Buf);
        clReleaseMemObject(c42Buf);

        clReleaseMemObject(c51Buf);
        clReleaseMemObject(c52Buf);

        clReleaseMemObject(w21sBuffer);
        clReleaseMemObject(w22sBuffer);

        clReleaseMemObject(b11sBuffer);
        clReleaseMemObject(b12sBuffer);
        clReleaseMemObject(b21sBuffer);
        clReleaseMemObject(b22sBuffer);

        clReleaseMemObject(icsBuf);
        clReleaseMemObject(ocsBuf);
        clReleaseMemObject(cscBuf);
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

    unsigned write_weight_sums(double* w21sptr, double* w22sptr, double* b11sptr, double* b12sptr, double* b21sptr, double* b22sptr)
    {
        w21sBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * k1 * k1 * sizeof(double),
                                   w21sptr,
                                   NULL);

        w22sBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * k1 * k1 * sizeof(double),
                                   w22sptr,
                                   NULL);

        b11sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b11sptr,
                                    NULL);

        b12sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b12sptr,
                                    NULL);

        b21sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b21sptr,
                                    NULL);

        b22sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b22sptr,
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

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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
        cs_compare(icsbuffer, ocsbuffer, cscBuf, ow, oh, od, cscInd);

        return 1;
    }

    unsigned relu(cl_mem ibuf, cl_mem obuf, int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(int), &olm);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

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

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned cs_compare(cl_mem ibuf, cl_mem obuf, cl_mem csbuf, int ow, int oh, int od, int csInd)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *) &csbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 3, sizeof(int), &csInd);

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

static void forward() {

    /*printf("l0 \n");
    for (int i = 0; i < (c1d); i++) {
        for (int j = 0; j < (c1h); j++) {
            for (int k = 0; k < (c1w); k++) {
                printf("%f ", matrixL0double[(i * c1h * c1w) + (j * c1w) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }*/

    /*for (int h = 0; h < (layer0d); h++) {
        for (int i = 0; i < (c1d); i++) {
            for (int j = 0; j < (k1); j++) {
                for (int k = 0; k < (k1); k++) {
                    printf("%f ", matrixW11double[(h * c1d * k1 * k1) + (i * k1 * k1) + (j * k1) + k]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }*/

    //conv block 1
    //convolution 1-1
    ocl.convolution3(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c11Buf,
                     c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d);
    //convolution 1-2
    ocl.convolution3(ocl.c11Buf, ocl.w12Buffer, ocl.b12Buffer, ocl.c12Buf,
                     c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d);
    //max pool 1
    ocl.maxpool(ocl.c12Buf, ocl.c21Buf, c1w, c1h, c1d, 2, 4, c2w, c2h);

    //conv block 2
    //convolution 2-1
    ocl.convolution3(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                     c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d);
    //convolution 2-2
    ocl.convolution3(ocl.c22Buf, ocl.w22Buffer, ocl.b22Buffer, ocl.c21Buf,
                     c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d);


    //max pool 2

    //convolution 3-1

    //convolution 3-2

    //max pool 3

    ocl.buf_read(c1w, c1h, c1d, matrixR, ocl.c12Buf);
    printf("matrixR \n");
    for (int i=0; i < (c1d); i++) {
        for (int j=0; j < (c1h); j++) {
            for (int k=0; k< (c1w); k++) {
                printf("%f ", matrixR[(i * c1h * c1w) + (j * c1w) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //max pool 2
    ocl.maxpool(ocl.c21Buf, ocl.c31Buf, c2w, c2h, c2d, 2, 4, c3w, c3h);



    //with 0 bias
    /*ocl.convolution3(ocl.l0Buffer, ocl.w01Buffer, nullptr, ocl.l1Buffer,
                     layer0w, layer0h, layer0d, k01, layer1w, layer1h, layer1d);*/

    ocl.buf_read(c2w, c2h, c2d, matrixR2, ocl.c22Buf);
    printf("matrixR2 \n");
    for (int i=0; i < (c2d); i++) {
        for (int j=0; j < (c2h); j++) {
            for (int k=0; k< (c2w); k++) {
                printf("%f ", matrixR2[(i * c2h * c2w) + (j * c2w) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }


    for (int i=0; i <10 ; i++) {
        printf("%f ", matrixR2[i]);
    }

}

int forward_abft() {
    int abftflag = 0;

    //conv block 1
    //convolution 1-1
    ocl.convolution3_abft(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c11Buf,
                     ocl.icsBuf, ocl.w11Buffer, ocl.b11sBuffer, ocl.ocsBuf,
                     c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d, 0);
    //convolution 1-2
    ocl.convolution3(ocl.c11Buf, ocl.w12Buffer, ocl.b12Buffer, ocl.c12Buf,
                     c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d);
    //max pool 1
    ocl.maxpool(ocl.c12Buf, ocl.c21Buf, c1w, c1h, c1d, 2, 4, c2w, c2h);

    //conv block 2
    //convolution 2-1
    ocl.convolution3(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                     c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d);
    //convolution 2-2
    ocl.convolution3_abft(ocl.c22Buf, ocl.w22Buffer, ocl.b22Buffer, ocl.c21Buf,
                          ocl.icsBuf, ocl.w22sBuffer, ocl.b22sBuffer, ocl. ocsBuf,
                          c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d, 4);


    ocl.buf_read(c2w, c2h, 1, ics, ocl.icsBuf);
    ocl.buf_read(c2w, c2h, 1, ocs, ocl.ocsBuf);

    printf("ics: \n ");
    for (int i=0; i <10 ; i++) {
        printf("%f ", ics[i]);
    }
    printf("\n");

    printf("ocs: \n ");
    for (int i=0; i <10 ; i++) {
        printf("%f ", ocs[i]);
    }
    printf("\n");

    ocl.buf_read(1, 1, 32, csc, ocl.cscBuf);
    printf("csc: \n ");
    for (int i=0; i <10 ; i++) {
        printf("%f ", csc[i]);
    }
    printf("\n");

    ocl.buf_read(c1w, c1h, c1d, matrixR, ocl.c12Buf);
    printf("matrixR \n");
    for (int i=0; i < (c1d); i++) {
        for (int j=0; j < (c1h); j++) {
            for (int k=0; k< (c1w); k++) {
                printf("%f ", matrixR[(i * c1h * c1w) + (j * c1w) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    //max pool 2
    ocl.maxpool(ocl.c21Buf, ocl.c31Buf, c2w, c2h, c2d, 2, 4, c3w, c3h);



    //with 0 bias
    /*ocl.convolution3(ocl.l0Buffer, ocl.w01Buffer, nullptr, ocl.l1Buffer,
                     layer0w, layer0h, layer0d, k01, layer1w, layer1h, layer1d);*/

    ocl.buf_read(c2w, c2h, c2d, matrixR2, ocl.c22Buf);
    printf("matrixR2 \n");
    for (int i=0; i < (c2d); i++) {
        for (int j=0; j < (c2h); j++) {
            for (int k=0; k< (c2w); k++) {
                printf("%f ", matrixR2[(i * c2h * c2w) + (j * c2w) + k]);
            }
            printf("\n");
        }
        printf("\n");
    }


    for (int i=0; i <10 ; i++) {
        printf("%f ", matrixR2[i]);
    }


    return abftflag;
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
    create_weight_sums();

    load_image("../../source-img/in0.png");

    for (int i=0; i <(layer0d * layer0h * layer0w) ; i++) {
        matrixL0double[i] = 1;
    }

    ocl.write_image(matrixL0double);

    ocl.create_layers();
    ocl.write_weights(matrixW11double, matrixW12double, matrixW21double, matrixW22double);
    ocl.write_bias(matrixB11double, matrixB12double, matrixB21double, matrixB22double);

    ocl.create_bufs_abft();
    ocl.write_weight_sums( matrixW21sum, matrixW22sum, matrixB11sum, matrixB12sum, matrixB21sum, matrixB22sum);

    forward();

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    //cleaning bufs and memory allocation
    freememory();

    //print opencl information
    printPlatformInfo(false);

    return 0;
}
