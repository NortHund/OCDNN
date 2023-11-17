#include "phase2.h"

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

int* matrixA;
int* matrixB;
int* matrixR;

short* matrixAshort;
short* matrixBshort;
short* matrixRshort;

char* matrixAchar;
char* matrixBchar;
char* matrixRchar;

double* matrixAdouble;
double* matrixBdouble;
double* matrixRdouble;

float* matrixAd;
float* matrixBd;
float* matrixRd;

float* matrixL0f;
float* matrixW01f;
float* matrixL1f;

double* matrixL0double;
double* matrixW01double;
double* matrixL1double;

double* ocs;
int ocsSize = 100000;


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
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/convolution-db.cl");
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

    unsigned mm_int()
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
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

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
                                     matrixR,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned mm_int_cs()
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

        cl_mem icsBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        sizeof(int),
                                        NULL,
                                        NULL);

        cl_mem ocsBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        sizeof(int),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(8), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 2, sizeof(cl_mem), (void *)&rBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 3, sizeof(cl_mem), (void *)&icsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 4, sizeof(cl_mem), (void *)&ocsBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(8),
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
                                     matrixR,
                                     0,
                                     NULL,
                                     NULL);

        int* ics;
        ics = (int*)malloc(sizeof(int));
        ics[0] = 0;

        int* ocs;
        ocs = (int*)malloc(sizeof(int));
        ocs[0] = 0;

        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     icsBuffer,
                                     0,
                                     0,
                                     sizeof(int),
                                     ics,
                                     0,
                                     NULL,
                                     NULL);

        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     ocsBuffer,
                                     0,
                                     0,
                                     sizeof(int),
                                     ocs,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        printf("int mm ics: %d, ocs: %d \n", ics[0], ocs[0]);

        free(ics);
        free(ocs);

        return (unsigned)status;
    }

    unsigned mm_short()
    {
        //Creating OpenCL buffers for matrices
        cl_mem aBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(short),
                                        matrixAshort,
                                        NULL);

        cl_mem bBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(short),
                                        matrixBshort,
                                        NULL);

        cl_mem rBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        matwidth * matheight * sizeof(short),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&rBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

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

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //Reading result from GPU memory to main memory
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     rBuffer,
                                     0,
                                     0,
                                     matwidth * matheight * sizeof(short),
                                     matrixRshort,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned mm_char()
    {
        //Creating OpenCL buffers for matrices
        cl_mem aBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(char),
                                        matrixAchar,
                                        NULL);

        cl_mem bBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(char),
                                        matrixBchar,
                                        NULL);

        cl_mem rBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        matwidth * matheight * sizeof(char),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 2, sizeof(cl_mem), (void *)&rBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(5),
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
                                     matwidth * matheight * sizeof(char),
                                     matrixRchar,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned mm_double()
    {
        //Creating OpenCL buffers for matrices
        cl_mem aBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(double),
                                        matrixAdouble,
                                        NULL);

        cl_mem bBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        matwidth * matheight * sizeof(double),
                                        matrixBdouble,
                                        NULL);

        cl_mem rBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        matwidth * matheight * sizeof(double),
                                        NULL,
                                        NULL);

        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *)&rBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(6),
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
                                     rBuffer,
                                     0,
                                     0,
                                     matwidth * matheight * sizeof(double),
                                     matrixRdouble,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned mm_float()
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
        global_work_size[0] = matwidth;
        global_work_size[1] = matheight;

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
                                     matrixRd,
                                     0,
                                     NULL,
                                     NULL);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
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

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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

    unsigned convolution_double_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);







        return (unsigned)status;
    }

    unsigned convolution_double_ocs_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {
        ocsBuffer = clCreateBuffer(_ocl_base->context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          (ocsSize) * sizeof(double),
                                          ocs,
                                          NULL);


    }

    unsigned convolution_double_ocs_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, int ocsInd)
    {
        //Reading result from GPU memory to main memory
        cl_int status;
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     ocsBuffer,
                                     0,
                                     0,
                                     (ocsSize) * sizeof(double),
                                     ocs,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //printf("Convolution-double ocl ocs: %f \n", ocs[ocsInd]);
    }


    double convolution_double_ics(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {

        double* ics;
        ics = (double*)malloc(sizeof(double));
        ics[0] = 0;

        cl_mem icsBuffer = clCreateBuffer(_ocl_base->context,
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          sizeof(double),
                                          ics,
                                          NULL);

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

        //Reading result from GPU memory to main memory
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     icsBuffer,
                                     0,
                                     0,
                                     sizeof(double),
                                     ics,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        printf("Convolution-double ocl ics: %f \n", ics[0]);

        double checksum = ics[0];

        free(ics);

        return checksum;
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
                                 matrixL0f,
                                 NULL);

        wBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 id * od * ww * wh * sizeof(float),
                                 matrixW01f,
                                 NULL);

        oBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 od * ow * oh * sizeof(float),
                                 matrixL1f,
                                 NULL);
    }

    unsigned convolution_read(int ow, int oh, int od) {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            oBuffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(float),
                                            matrixL1f,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution_ocs(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        float* l1oc;
        l1oc = (float*)malloc(sizeof(float));
        l1oc[0] = 1;

        cl_mem ocBuffer = clCreateBuffer(_ocl_base->context,
                                         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         sizeof(float),
                                         l1oc,
                                         NULL);

        cl_mem cBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        sizeof(float),
                                        NULL,
                                        NULL);

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&oBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&ocBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(int), &oh);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 3, sizeof(int), &ow);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 4, sizeof(int), &od);

        size_t global_work_size[1];
        global_work_size[0] = 1;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(3),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        float out = 0;



        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     ocBuffer,
                                     0,
                                     0,
                                     sizeof(float),
                                     l1oc,
                                     0,
                                     NULL,
                                     &_event);

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        printf("ocl checksum: %f\n", l1oc[0]);

        free(l1oc);

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
                        wSum += matrixW01f[(n * od * wh * ww) + (m * wh * ww) + (i * ww) + j];
                    }
                    //printf("wSum: %f\n", wSum);
                    xSum = 0;
                    for (int r = 0; r < oh; ++r) {
                        for (int c = 0; c < ow; ++c) {
                            xSum += matrixL0f[(n * ih * iw) + ((r + i) * iw) + (c + j)];
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

        for (int d = 0; d < layer1d; d++) {
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    checksum += matrixL1f[(d * layer1h * layer1w) + (i * layer1w) + j];
                }
            }
        }
        printf("ouput checksum: %f \n", checksum);

        return checksum;
    }

    int mm_int_ics(int iw, int ih, int ow, int oh, int id)
    {
        int checksum = 0;
        int sumRow[iw];
        int sumCol[oh];

        for (int i = 0; i < iw; i++) {
            sumRow[i] = 0;
            for (int j = 0; j < ih; j++) {
                sumRow[i] += matrixA[(j * iw) + i];
            }
        }

        for (int i = 0; i < oh; i++) {
            sumCol[i] = 0;
            for (int j = 0; j < ow; j++) {
                sumCol[i] += matrixB[(i * ow) + j];
            }
        }

        for (int i = 0; i < oh; i++) {
            checksum += sumRow[i] * sumCol[i];
        }

        printf("mm_int input checksum: %d \n", checksum);

        return checksum;
    }

    int mm_int_ocs(int iw, int ih, int ow, int oh, int id)
    {
        int checksum = 0;
        int sumRow[iw];
        int sumCol[oh];

        for (int i = 0; i < iw; i++) {
            for (int j = 0; j < ih; j++) {
                checksum += matrixR[(i * iw) + j];
            }
        }

        printf("mm_int output checksum: %d \n", checksum);

        return checksum;
    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times\n\n";
        std::cout << "  Matrix addition: " << kernel_execution_times[0] << " us\n";
        std::cout << "  Mm_float: " << kernel_execution_times[1] << " us\n";
        std::cout << "  Convolution fl: " << kernel_execution_times[2] << " us\n\n";
    }

    std::unique_ptr<OCL_Base> _ocl_base;

    cl_mem iBuffer = nullptr;
    cl_mem wBuffer = nullptr;

    cl_mem iBufferd = nullptr;
    cl_mem wBufferd = nullptr;
    cl_mem oBuffer = nullptr;
    cl_mem oBufferd = nullptr;
    cl_mem ocsBuffer = nullptr;



private:
    cl_program prog_mm_int ;
    cl_program prog_mm_float;
    cl_program prog_cv_d;
    cl_program prog_cv_oc;




    cl_event _event;

    // 0 - Matrix addition
    // 1 - Matrix addition float
    // 5 - convolution
    unsigned long kernel_execution_times[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

OCL_Phase2 ocl_phase2;

struct CreateMatrices : public IProgram
{
    int run() override
    {

        matrixA = (int*)malloc((matwidth * matheight) * sizeof(int));
        matrixB = (int*)malloc((matwidth * matheight) * sizeof(int));
        matrixR = (int*)malloc((matwidth * matheight) * sizeof(int));

        for (int i=0; i<matheight; i++) {
            for (int j=0; j<matwidth; j++) {
                matrixA[i * matwidth + j] = i + 1;
                matrixB[i * matwidth + j] = j + 2;
            }
        }

        matrixAd = (float*)malloc((matwidth * matheight) * sizeof(float));
        matrixBd = (float*)malloc((matwidth * matheight) * sizeof(float));
        matrixRd = (float*)malloc((matwidth * matheight) * sizeof(float));

        for (int i=0; i<matheight; i++) {
            for (int j=0; j<matwidth; j++) {
                matrixAd[i * matwidth + j] = i + 1;
                matrixBd[i * matwidth + j] = j + 2;
            }
        }

        matrixAdouble = (double*)malloc((matwidth * matheight) * sizeof(double));
        matrixBdouble = (double*)malloc((matwidth * matheight) * sizeof(double));
        matrixRdouble = (double*)malloc((matwidth * matheight) * sizeof(double));

        ocs = (double*)malloc((ocsSize) * sizeof(double));

        for (int i=0; i<matheight; i++) {
            for (int j=0; j<matwidth; j++) {
                matrixAdouble[i * matwidth + j] = i + 1;
                matrixBdouble[i * matwidth + j] = j + 2;
            }
        }

        matrixAshort = (short*)malloc((matwidth * matheight) * sizeof(short));
        matrixBshort = (short*)malloc((matwidth * matheight) * sizeof(short));
        matrixRshort = (short*)malloc((matwidth * matheight) * sizeof(short));

        for (short i=0; i<matheight; i++) {
            for (short j=0; j<matwidth; j++) {
                matrixAshort[i * matwidth + j] = i + 1;
                matrixBshort[i * matwidth + j] = j + 2;
            }
        }

        matrixAchar = (char*)malloc((matwidth * matheight) * sizeof(char));
        matrixBchar = (char*)malloc((matwidth * matheight) * sizeof(char));
        matrixRchar = (char*)malloc((matwidth * matheight) * sizeof(char));

        for (char i=0; i<matheight; i++) {
            for (char j=0; j<matwidth; j++) {
                matrixAchar[i * matwidth + j] = i + 1;
                matrixBchar[i * matwidth + j] = j + 2;
            }
        }

        matrixL0f = (float*)malloc((layer0d) * (layer0w * layer0h) * sizeof(float));
        matrixW01f = (float*)malloc((layer0d) * (layer1d) * (w01w * w01h) * sizeof(float));
        matrixL1f = (float*)malloc((layer1d) * (layer1w * layer1h) * sizeof(float));

        for (int i=0; i<layer0h; i++) {
            for (int j=0; j<layer0w; j++) {
                matrixL0f[i * layer0w + j] = 2.5;
            }
        }

        for (int n = 0; n < layer0d; n++) {
            for (int d = 0; d < layer1d; d++) {
                for (int i = 0; i < w01h; i++) {
                    for (int j = 0; j < w01w; j++) {
                        matrixW01f[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + (i * w01w) + j] = 0.01 + 0.01 * d;
                    }
                }
            }
        }

        for (int d = 0; d < layer1d; d++) {
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    matrixL1f[(d * layer1h * layer1w) + (i * layer1w) + j] = d;
                }
            }
        }

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
        //ocl_phase2.mm_int_cs();
        //ocl_phase2.mm_int_ocs(matwidth, matheight, matwidth, matheight, 1);

        ocl_phase2.mm_int_ics(matwidth, matheight, matwidth, matheight, 1);
        ocl_phase2.mm_int();
        ocl_phase2.mm_int_ocs(matwidth, matheight, matwidth, matheight, 1);

        ocl_phase2.mm_float();
        ocl_phase2.mm_short();
        ocl_phase2.mm_char();
        ocl_phase2.mm_double();

        ocl_phase2.convolution_input_checksum(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        ocl_phase2.convolution_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 1);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 2);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 3);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 4);
        ocl_phase2.convolution_fl(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 5);
        ocl_phase2.convolution_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 5);
        ocl_phase2.convolution_read(layer1w, layer1h, layer1d);

        ocl_phase2.convolution_output_checksum(layer1w, layer1h, layer1d);


        ocl_phase2.convolution_double_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

        double doubleics = 0;
        doubleics = ocl_phase2.convolution_double_ics(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
        printf("double ics %f \n", doubleics);


        ocl_phase2.convolution_double_ocs_write(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 2);
        for (int i = 0; i < ocsSize; i++) {
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 1);
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 2);
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 3);
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 4);
            ocl_phase2.convolution_double(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 5);

            ocl_phase2.convolution_double_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
            ocl_phase2.convolution_double_ocs(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, i);
        }
        ocl_phase2.convolution_double_ocs_read(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, 2);
        for (int i = 0; i < ocsSize; i++) {
            printf("ocs %d: %f \n", i, ocs[i]);
            if (abs(doubleics - ocs[i]) > 0.00000001) {
                printf("checksum mismatch! \n");
            }
        }


        unsigned error = 0;
        return (int) error;
    }
};

struct SaveMatrixOCL : public IProgram
{
    int run() override {

        FILE *fp = fopen("../../output-data/matA.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%d ", matrixA[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matB.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%d ", matrixB[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/p2-mm-int32.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%d ", matrixR[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/p2-mm-int16.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%d ", matrixRshort[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/p2-mm-int8.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%d ", matrixRchar[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/p1-mm-f32.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%f ", matrixRd[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/p1-mm-f64.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < matheight; i++) {
            for (int j = 0; j < matwidth; j++) {
                fprintf(fp, "%f ", matrixRdouble[i * matwidth + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matrixL1f.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int d = 0; d < layer1d; d++) {
            fprintf(fp, "//M = %d \n", d);
            for (int i = 0; i < layer1h; i++) {
                for (int j = 0; j < layer1w; j++) {
                    fprintf(fp, "%f ", matrixL1f[(d * layer1h * layer1w) + i * layer1w + j]);
                }
                fprintf(fp, "\n");
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matrixL0f.txt", "w");
        if (fp == NULL) {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i = 0; i < layer0h; i++) {
            for (int j = 0; j < layer0w; j++) {
                fprintf(fp, "%f ", matrixL0f[i * layer0w + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        fp = fopen("../../output-data/matrixW01f.txt", "w");
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
                        fprintf(fp, "%f ", matrixW01f[(n * layer1d * w01h * w01w) + (d * w01h * w01w) + i * w01w + j]);
                    }
                    fprintf(fp, "\n");
                }
                fprintf(fp, "\n");
            }
        }
        fclose(fp);

        fp = fopen("../../output-data/matrixL1d.txt", "w");
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

    free(matrixA);
    free(matrixB);
    free(matrixR);

    free(matrixAd);
    free(matrixBd);
    free(matrixRd);

    free(matrixAshort);
    free(matrixBshort);
    free(matrixRshort);

    free(matrixAchar);
    free(matrixBchar);
    free(matrixRchar);

    free(matrixL0f);
    free(matrixW01f);
    free(matrixL1f);

    free(matrixL0double);
    free(matrixW01double);
    free(matrixL1double);

    free(matrixAdouble);
    free(matrixBdouble);
    free(matrixRdouble);

    free(ocs);

    printPlatformInfo(false);
    return 0;
}