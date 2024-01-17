#include "phase6_model.h"

#define OUTPUT = 1000;

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
        prog_util = _ocl_base->CreateProgramFromFile("kernels/p5-util.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_double"); //0
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "input_sum"); //1
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "output_sum"); //2
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "cs_compare"); //3
        _ocl_base->CreateKernelFromProgram(prog_util, "relu"); //4
        _ocl_base->CreateKernelFromProgram(prog_util, "maxpool"); //5
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat"); //6
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat_ics"); //7
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

    unsigned setbufs_l01() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l0Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w01Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l1Buffer);
    }

    unsigned setbuf_l1rb()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l1Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l1rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b01Buffer);
    }

    unsigned setbuf_l1rbd()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l1Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l1rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b01Buffer);
    }

    unsigned setbuf_l12()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)&l1rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)&l2Buffer);
    }

    unsigned setbuf_l12d()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)&l1rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)&l2dBuffer);
    }

    unsigned setbufs_l23() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l2Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w23Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l3Buffer);
    }

    unsigned setbuf_l34()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)&l3rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)&l4Buffer);
    }

    unsigned setbuf_l3rb()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l3Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l3rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b23Buffer);
    }

    unsigned setbuf_l3rbd()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l3Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l3rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b23Buffer);
    }

    unsigned setbuf_l4d()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)&l3rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)&l4dBuffer);
    }

    unsigned setbufs_l45() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l4Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w45Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l5Buffer);
    }

    unsigned setbuf_l5rb()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l5Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l5rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b45Buffer);
    }

    unsigned setbuf_l5rbd()
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&l5Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&l5rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&b45Buffer);
    }


    unsigned setbufs_l01ics() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l0Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w01sumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l1insumBuffer);
    }

    unsigned setbufs_l23ics() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l2Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w23sumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l3insumBuffer);
    }

    unsigned setbufs_l45ics() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &l4Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &w45sumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &l5insumBuffer);
    }

    unsigned setbufs_l01ocs() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&l1Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&l1outsumBuffer);
    }

    unsigned setbufs_l23ocs() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&l3Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&l3outsumBuffer);
    }

    unsigned setbufs_l45ocs() {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&l5Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&l5outsumBuffer);
    }

    unsigned setbufs_l01csc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l1insumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l1outsumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l1rbcsc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l1rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l1rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l2csc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l2Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l2dBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l23csc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l3insumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l3outsumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l3rbcsc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l3rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l3rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l4csc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l4Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l4dBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l45csc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l5insumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l5outsumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned setbufs_l5rbcsc() {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&l5rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&l5rbdBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&cscBuffer);
    }

    unsigned zero_bufs()
    {
        cl_int status;

    }

    unsigned free_bufs()
    {
        clReleaseMemObject(iBuffer);
        clReleaseMemObject(wBuffer);
        clReleaseMemObject(oBuffer);
        clReleaseMemObject(ocsBuffer);
        clReleaseMemObject(icsBuffer);
        clReleaseMemObject(cscBuffer);
        clReleaseMemObject(biasBuffer);

        clReleaseMemObject(l0Buffer);
        clReleaseMemObject(l1Buffer);
        clReleaseMemObject(l1rbBuffer);
        clReleaseMemObject(l2Buffer);
        clReleaseMemObject(l3Buffer);
        clReleaseMemObject(l3rbBuffer);
        clReleaseMemObject(l4Buffer);
        clReleaseMemObject(l5Buffer);
        clReleaseMemObject(l5rbBuffer);
        clReleaseMemObject(l6Buffer);

        clReleaseMemObject(b01Buffer);
        clReleaseMemObject(b23Buffer);
        clReleaseMemObject(b45Buffer);
        clReleaseMemObject(b56Buffer);

        clReleaseMemObject(w01Buffer);
        clReleaseMemObject(w23Buffer);
        clReleaseMemObject(w45Buffer);
        clReleaseMemObject(w56Buffer);

        clReleaseMemObject(l1insumBuffer);
        clReleaseMemObject(l3insumBuffer);
        clReleaseMemObject(l5insumBuffer);
        clReleaseMemObject(l6insumBuffer);

        clReleaseMemObject(l1outsumBuffer);
        clReleaseMemObject(l3outsumBuffer);
        clReleaseMemObject(l5outsumBuffer);

        clReleaseMemObject(w01sumBuffer);
        clReleaseMemObject(w23sumBuffer);
        clReleaseMemObject(w45sumBuffer);
        clReleaseMemObject(w56sumBuffer);
    }

    double buf_read(int ow, int oh, int od, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            l1Buffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);
    }

    double last_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            l6Buffer,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned convolution_nb(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;
        //Setting kernel arguments
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

    unsigned relu_nb(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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

    unsigned maxpool_nb(int iw, int ih, int id, int stride, int kernel_size, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(5), 2, sizeof(int), &olm);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 3, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 4, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 5, sizeof(int), &kernel_size);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 6, sizeof(int), &stride);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

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
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            std::cerr << "At d: " << olm  << std::endl;
            exit(EXIT_FAILURE);
        }

        kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned output_sum_nb(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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

    unsigned cs_compare_nb(int layer, int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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

    double cs_compare_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            cscBuffer,
                                            0,
                                            0,
                                            10 * sizeof(double),
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

        /*printf("csc: \n");
        for (int i = 0; i < 5; i++) {
            printf("%f ", csc[i]);
        }
        printf("\n");*/
    }

    unsigned setbuf_l56ics()
    {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(7), 0, sizeof(cl_mem), (void *)&l5rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 1, sizeof(cl_mem), (void *)&l6insumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 2, sizeof(cl_mem), (void *)&w56sumBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 3, sizeof(cl_mem), (void *)&b56Buffer);
    }

    unsigned setbuf_l56()
    {
        cl_int status;
        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *)&l5rbBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *)&l6Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *)&w56Buffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 3, sizeof(cl_mem), (void *)&b56Buffer);
    }

    unsigned flatmat_nb(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
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

    double flatmat_ics_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            l6insumBuffer,
                                            0,
                                            0,
                                            sizeof(double),
                                            optr,
                                            0,
                                            NULL,
                                            &_event);

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned flatmat(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *)&iBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *)&oBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *)&wBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 3, sizeof(cl_mem), (void *)&biasBuffer);
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

    unsigned flatmat_write(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* iptr, double* wptr, double* bptr)
    {
        iBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 id * iw * ih * sizeof(double),
                                 iptr,
                                 NULL);


        oBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_WRITE,
                                 od * ow * oh * sizeof(double),
                                 NULL,
                                 NULL);

        wBuffer = clCreateBuffer(_ocl_base->context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 id * iw * ih * ow * sizeof(double),
                                 wptr,
                                 NULL);

        biasBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    ow * sizeof(double),
                                    bptr,
                                    NULL);
    }

    double flatmat_read(int iw, int ih, int id, int ww, int wh, int ow, int oh, int od, int iln, int olm, double* optr)
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

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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
    cl_mem biasBuffer = nullptr;

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
    cl_program prog_cv_d;
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

OCL_Phase2 ocl_phase2;