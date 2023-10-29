#include "phase1.h"

int scaling_factor = 4;

int matwidth = 100;
int matheight = 100;

int mfwidth = 10;
int mfheight = 10;

int layer0w = 32;
int layer0h = 32;
int layer0d = 1;

int layer1w = 28;
int layer1h = 28;
int layer1d = 6;

int w01w = 5;
int w01h = 5;

int* matrixA;
int* matrixB;
int* matrixRcpp;
int* matrixRocl;

float* matrixAd;
float* matrixBd;
float* matrixRocld;

float* matrixAf;
float* matrixBf;
float* matrixRoclf;

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

class OCL_Phase1
{
public:
    OCL_Phase1()
    {
        _ocl_base.reset(new OCL_Base());

        init_programs();
        init_kernels();
    }

    ~OCL_Phase1()
    {
    }

    void init_programs()
    {
        prog_ma = _ocl_base->CreateProgramFromFile("kernels/p1-ma.cl");
        prog_mad = _ocl_base->CreateProgramFromFile("kernels/mad.cl");
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/convolution-db.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_ma, "matrix_addition");
        _ocl_base->CreateKernelFromProgram(prog_mad, "mad");
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_fl");
    }

    unsigned matrix_addition()
    {
        //Creating OpenCL buffers for matrices
        cl_mem aBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(int),
                                        matrixA,
                                        NULL);

        cl_mem bBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(int),
                                        matrixB,
                                        NULL);

        cl_mem rBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        matwidth * matheight * sizeof(int),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *)&rBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth * sizeof(int);
        global_work_size[1] = matheight * sizeof(int);

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

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //Reading result from GPU memory to main memory
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                            rBuffer,
                            0,
                            0,
                            matwidth * matheight * sizeof(int),
                            matrixRocl,
                            0,
                            NULL,
                            NULL);

        return (unsigned)status;
    }

    unsigned mad()
    {
        //Creating OpenCL buffers for matrices
        cl_mem aBufferd = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(float),
                                        matrixAd,
                                        NULL);

        cl_mem bBufferd = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(float),
                                        matrixBd,
                                        NULL);

        cl_mem rBufferd = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        matwidth * matheight * sizeof(float),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(1), 0, sizeof(cl_mem), (void *)&aBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 1, sizeof(cl_mem), (void *)&bBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 2, sizeof(cl_mem), (void *)&rBufferd);

        size_t global_work_size[2];
        global_work_size[0] = matwidth * sizeof(float);
        global_work_size[1] = matheight * sizeof(float);

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

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //Reading result from GPU memory to main memory
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     rBufferd,
                                     0,
                                     0,
                                     matwidth * matheight * sizeof(float),
                                     matrixRocld,
                                     0,
                                     NULL,
                                     NULL);

        return (unsigned)status;
    }

    unsigned convolution_fl(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&iBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&wBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 2, sizeof(cl_mem), (void *)&oBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 3, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 4, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 5, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 6, sizeof(int), &wh);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 7, sizeof(int), &ww);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 8, sizeof(int), &od);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 9, sizeof(int), &iln);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 10, sizeof(int), &olm);

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

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od) {
        //Creating OpenCL buffers for matrices
        iBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        iw * ih * sizeof(float),
                                        matrixAf,
                                        NULL);

        wBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        id * od * ww * wh * sizeof(float),
                                        matrixBf,
                                        NULL);

        oBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        od * ow * oh * sizeof(float),
                                        matrixRoclf,
                                        NULL);
    }

    unsigned convolution_read(int ow, int oh, int od) {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     oBuffer,
                                     0,
                                     0,
                                     od * ow * oh * sizeof(float),
                                     matrixRoclf,
                                     0,
                                     NULL,
                                     NULL);
        return (unsigned)status;
    }

    float convolution_input_checksum(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        float wSum = 0;
        float xSum = 0;
        float checksum = 0;
        for (int n = 0; n < id; ++n) {
            for (int i = 0; i < wh; ++i) {
                for (int j = 0; j < ww; ++j) {
                    wSum = 0;
                    for (int m = 0; m < od; ++m) {
                        wSum += matrixBf[(n * od * wh * ww) + (m * wh * ww) + (i * ww) + j];
                    }
                    //printf("wSum: %f\n", wSum);
                    xSum = 0;
                    for (int r = 0; r < oh; ++r) {
                        for (int c = 0; c < ow; ++c) {
                            xSum += matrixAf[(n * ih * iw) + ((r + i) * iw) + (c + j)];
                        }
                    }
                    //printf("xSum: %f\n", xSum);
                    checksum += xSum * wSum;
                    //printf("n: %d, i: %d, j: %d\n checksum: %f, xSum: %f, wSum: %f\n", n, i, j, checksum, xSum, wSum);
                }
            }
        }

        printf("input checksum: %f\n", checksum);
        return checksum;
    }

    float convolution_output_checksum(int ow, int oh, int od)
    {
        float checksum = 0;
        int madmax = 0;
        int count = 0;
        madmax = ow * oh * od;
        for (int i = 0; i < madmax; ++i) {
                checksum = checksum + matrixRoclf[i];
                count++;
                //printf("value : %f, output checksum: %f\n", matrixRoclf[i], checksum);
        }
        //printf("ouput checksum: %f, count: %d \n", checksum, count);

        checksum = 0;

        for (int d = 0; d < layer1d; d++) {
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    checksum += matrixRoclf[(d * layer1h * layer1w) + (i * layer1w) + j];
                }
            }
        }
        printf("ouput checksum: %f \n", checksum);

        return checksum;
    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times\n\n";
        std::cout << "  Matrix addition: " << kernel_execution_times[0] << " us\n";
        std::cout << "  Mad: " << kernel_execution_times[1] << " us\n";
        std::cout << "  Convolution fl: " << kernel_execution_times[2] << " us\n\n";
    }

    std::unique_ptr<OCL_Base> _ocl_base;

    cl_mem iBuffer = nullptr;
    cl_mem wBuffer = nullptr;
    cl_mem oBuffer = nullptr;

private:
    cl_program prog_ma;
    cl_program prog_mad;
    cl_program prog_cv_d;

    cl_event _event;

    // 0 - Matrix addition
    // 1 - Matrix addition float - Mad
    // 5 - convolution
    unsigned long kernel_execution_times[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

OCL_Phase1 ocl_phase1;

    struct CreateMatrices : public IProgram
    {
        int run() override
        {

            matrixA = (int*)malloc((matwidth * matheight) * sizeof(int));
            matrixB = (int*)malloc((matwidth * matheight) * sizeof(int));
            matrixRcpp = (int*)malloc((matwidth * matheight) * sizeof(int));
            matrixRocl = (int*)malloc((matwidth * matheight) * sizeof(int));

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    matrixA[i * matwidth + j] = i + 1;
                    matrixB[i * matwidth + j] = j + 2;
                }
            }

            matrixAd = (float*)malloc((matwidth * matheight) * sizeof(float));
            matrixBd = (float*)malloc((matwidth * matheight) * sizeof(float));
            matrixRocld = (float*)malloc((matwidth * matheight) * sizeof(float));

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    matrixAd[i * matwidth + j] = i + 1;
                    matrixBd[i * matwidth + j] = j + 2;
                }
            }

            matrixAf = (float*)malloc((layer0d) * (layer0w * layer0h) * sizeof(float));
            matrixBf = (float*)malloc((layer0d) * (layer1d) * (w01w * w01h) * sizeof(float));
            matrixRoclf = (float*)malloc((layer1d) * (layer1w * layer1h) * sizeof(float));

            for (int i=0; i<layer0h; i++) {
                for (int j=0; j<layer0w; j++) {
                    matrixAf[i * layer0w + j] = 2.5;
                }
            }

            for (int n = 0; n < layer0d; n++) {
                for (int d = 0; d < layer1d; d++) {
                    for (int i = 0; i < w01h; i++) {
                        for (int j = 0; j < w01w; j++) {
                            matrixBf[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + (i * w01w) + j] = 0.01 + 0.01 * d;
                        }
                    }
                }
            }

            for (int d = 0; d < layer1d; d++) {
                for (int i = 0; i < layer1h; i++) {
                    for (int j = 0; j < layer1w; j++) {
                        matrixRoclf[(d * layer1h * layer1w) + (i * layer1w) + j] = d;
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
            ocl_phase1.matrix_addition();
            ocl_phase1.mad();

            ocl_phase1.convolution_input_checksum(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

            ocl_phase1.convolution_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 1);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 2);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 3);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 4);
            ocl_phase1.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 5);
            ocl_phase1.convolution_read(layer1w, layer1h, layer1d);

            ocl_phase1.convolution_output_checksum(layer1w, layer1h, layer1d);

            unsigned error = 0;
            return (int) error;
        }
    };

    struct SaveMatrixOCL : public IProgram
    {
        int run() override {

            FILE *fp = fopen("../../output-img/p1-matAddOcl.txt", "w");
            if (fp == NULL) {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i = 0; i < matheight; i++) {
                for (int j = 0; j < matwidth; j++) {
                    fprintf(fp, "%d ", matrixRocl[i * matwidth + j]);
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            fp = fopen("../../output-img/p1-madOcl.txt", "w");
            if (fp == NULL) {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i = 0; i < matheight; i++) {
                for (int j = 0; j < matwidth; j++) {
                    fprintf(fp, "%f ", matrixRocld[i * matwidth + j]);
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            fp = fopen("../../output-data/l1-conv-out.txt", "w");
            if (fp == NULL) {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int d = 0; d < layer1d; d++) {
                fprintf(fp, "//M = %d \n", d);
                for (int i = 0; i < layer1h; i++) {
                    for (int j = 0; j < layer1w; j++) {
                        fprintf(fp, "%f ", matrixRoclf[(d * layer1h * layer1w) + i * layer1w + j]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            fp = fopen("../../output-data/l0-matrixAf.txt", "w");
            if (fp == NULL) {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i = 0; i < layer0h; i++) {
                for (int j = 0; j < layer0w; j++) {
                    fprintf(fp, "%f ", matrixAf[i * layer0w + j]);
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            fp = fopen("../../output-data/l0-matrixBf.txt", "w");
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
                            fprintf(fp, "%f ", matrixBf[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + i * w01w + j]);
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

    //Step 1
    ProgramStopwatch Program_sw(clock);

    //Step 2
    CreateMatrices createMatrices;
    MatrixAdditionOCL matrixAdditionOCL;
    SaveMatrixOCL saveMatrixOCL;

    int result = 0;


    result = Program_sw.runProgram(createMatrices);
    std::cout << "Creation of matrices: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(matrixAdditionOCL);
    std::cout << "OpenCL matrix addition: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveMatrixOCL);
    std::cout << "OpenCL matrix save: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    //Step 4
    std::cout << "Step 4 - convolution" << std::endl;
    std::cout << "OpenCl: Save image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl_phase1.print_kernel_execution_times();

    free(matrixA);
    free(matrixB);
    free(matrixRcpp);
    free(matrixRocl);

    free(matrixAd);
    free(matrixBd);
    free(matrixRocld);

    free(matrixAf);
    free(matrixBf);
    free(matrixRoclf);

    printPlatformProfile(false);
    return 0;
}
