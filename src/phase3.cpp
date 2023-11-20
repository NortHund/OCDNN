#include "phase3.h"
int scaling_factor = 4;

int matwidth = 10;
int matheight = 10;

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
        std::cout << "  Input checksum read: " << kernel_execution_times[3] << " us\n";
        std::cout << "  Output checksum: " << kernel_execution_times[4] << " us\n";
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

        ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        ocl_phase2.convolution_double_ics_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_double_ics(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_double_ics_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        for (int i = 0; i < layer0d; i++) {
            for (int j = 0; j < layer1d; j++) {
                ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, i, j);
            }
        }
        ocl_phase2.convolution_double_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        ocl_phase2.convolution_double_ocs_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
        ocl_phase2.convolution_double_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);
        ocl_phase2.convolution_double_ocs_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 0);

        //printf("ocs %d: %f \n", i, ocs[i]);
        if (abs(ics[0] - ocs[0]) > 0.00000000001) {
            printf("checksum mismatch! ics: %.16f, ocs: %.16f \n", ics[0], ocs[0]);
        }
        printf("ics: %.16f, ocs: %.16f. \n", ics[0], ocs[0]);


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
