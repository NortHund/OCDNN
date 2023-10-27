#include "phase1.h"

lodepng_wrapper::LodepngWrapper img0;
//lodepng_wrapper::LodepngWrapper img1;

int scaling_factor = 4;
int avg_window_size = 5;

int matwidth = 100;
int matheight = 100;
int* matrixA;
int* matrixB;
int* matrixRcpp;
int* matrixRocl;

int* matrixW;

float* matrixWd;
float* inputIm;
float* layer1;

float* matrixAd;
float* matrixBd;
float* matrixRocld;

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

class OCL_image : public lodepng_wrapper::LodepngWrapper
{
public:
    OCL_image(std::unique_ptr<OCL_Base>& ocl_base)
        : _ocl_base(ocl_base)
    {
    }

    ~OCL_image()
    {
        if(imageBuffer != nullptr) clReleaseMemObject(imageBuffer);
        if(averageBuffer != nullptr) clReleaseMemObject(averageBuffer);
    }

    unsigned transform_to_grayscale() override
    {
        cl_mem inputBuffer = clCreateBuffer(_ocl_base->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            _width * _height * 4 * sizeof(unsigned char), _image, NULL);
        imageBuffer = clCreateBuffer(_ocl_base->context, CL_MEM_READ_WRITE,
                                            _width * _height * sizeof(unsigned char), NULL, NULL);

        clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *)&inputBuffer);
        clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *)&imageBuffer);

        size_t global_work_size[1] = {_width * _height * 4 * sizeof(unsigned char)};
        clEnqueueNDRangeKernel(_ocl_base->commandQueue, _ocl_base->GetKernel(0), 1, NULL,
                            global_work_size, NULL, 0, NULL, &_event);

        kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        clReleaseMemObject(inputBuffer);

        _imageIsGrayscaled = true;

        return 0;
    }

    void clone_image(unsigned char *dest) override
    {
        clEnqueueReadBuffer(_ocl_base->commandQueue, imageBuffer, CL_TRUE, 0,
                    _width * _height * sizeof(unsigned char), dest, 0, NULL, NULL);
    }

    unsigned resize_image(unsigned scalingFactor) override
    {
        _width = _width / scalingFactor;
        _height = _height / scalingFactor;

        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(1), 0, sizeof(scalingFactor), &scalingFactor);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 1, sizeof(cl_mem), (void *)&imageBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(1), 2, sizeof(cl_mem), (void *)&imageBuffer);

        size_t global_work_size[2];
        global_work_size[0] = _width * sizeof(unsigned char);
        global_work_size[1] = _height * sizeof(unsigned char);

        clEnqueueNDRangeKernel(_ocl_base->commandQueue, _ocl_base->GetKernel(1), 2, NULL,
                            global_work_size, NULL, 0, NULL, &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned average(int windowSize)
    {
        averageBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_WRITE,
                                        _width * _height * sizeof(unsigned char),
                                        NULL,
                                        NULL);


        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *)&imageBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *)&averageBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 2, sizeof(int), &windowSize);

        size_t global_work_size[2];
        global_work_size[0] = _width * sizeof(unsigned char);
        global_work_size[1] = _height * sizeof(unsigned char);

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

        status = clEnqueueCopyBuffer(_ocl_base->commandQueue,
                            averageBuffer,
                            imageBuffer,
                            0,
                            0,
                            _width * _height * sizeof(unsigned char),
                            0,
                            NULL,
                            NULL);

        return (unsigned)status;
    }

    unsigned convolution(int windowSize)
    {
        convolutionBuffer = clCreateBuffer(_ocl_base->context,
                                       CL_MEM_READ_WRITE,
                                       _width * _height * sizeof(unsigned char),
                                       NULL,
                                       NULL);

        cl_mem wBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        avg_window_size * avg_window_size * sizeof(int),
                                        matrixW,
                                        NULL);


        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *)&imageBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *)&convolutionBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 2, sizeof(cl_mem), (void *)&wBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 3, sizeof(int), &windowSize);

        size_t global_work_size[2];
        global_work_size[0] = _width * sizeof(unsigned char);
        global_work_size[1] = _height * sizeof(unsigned char);

        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(4),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);

        kernel_execution_times[2] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        status = clEnqueueCopyBuffer(_ocl_base->commandQueue,
                                     convolutionBuffer,
                                     imageBuffer,
                                     0,
                                     0,
                                     _width * _height * sizeof(unsigned char),
                                     0,
                                     NULL,
                                     NULL);

        return (unsigned)status;
    }

    unsigned save_image(const char* filename) override
    {
        if(_imageIsGrayscaled)
        {
            _image = (unsigned char *)realloc(_image, _width * _height * sizeof(unsigned char));
            clone_image(_image);
            return LodepngWrapper::save_image(filename);
        }
        else
        {
            return LodepngWrapper::save_image(filename);
        }
    }

    int get_width()
    {
        return _width;
    }

    int get_height()
    {
        return _height;
    }

    void print_kernel_execution_times()
    {
        std::cout << "    Grayscale kernel execution time: " << kernel_execution_times[0] << " us\n";
        std::cout << "    Resize kernel execution time: " << kernel_execution_times[1] << " us\n";
        std::cout << "    Average kernel execution time: " << kernel_execution_times[2] << " us\n";

        std::cout << "    Total: " << kernel_execution_times[0] 
                                                        + kernel_execution_times[1] 
                                                        + kernel_execution_times[2] 
                                                        + kernel_execution_times[3] << " us\n" << std::endl;
    }


    cl_mem imageBuffer = nullptr;
    cl_mem averageBuffer = nullptr;
    cl_mem stdDeviationBuffer = nullptr;
    cl_mem convolutionBuffer = nullptr;
private:
    
    std::unique_ptr<OCL_Base>& _ocl_base;

    cl_event _event;

    // 0 - grayscale
    // 1 - resize
    // 2 - average
    unsigned long kernel_execution_times[5] = {0, 0, 0, 0, 0};
};

class OCL_Phase1
{
public:
    OCL_Phase1()
    {
        _ocl_base.reset(new OCL_Base());

        img1.reset(new OCL_image(_ocl_base));
        
        init_programs();
        init_kernels();
    }

    ~OCL_Phase1()
    {
    }

    void init_programs()
    {
        prog = _ocl_base->CreateProgramFromFile("kernels/p1-4-5-grayscale-resize.cl");
        prog_average = _ocl_base->CreateProgramFromFile("kernels/p1-moving-avg.cl");
        prog_ma = _ocl_base->CreateProgramFromFile("kernels/p1-ma.cl");
        prog_cv = _ocl_base->CreateProgramFromFile("kernels/convolution-image.cl");
        prog_cv_d = _ocl_base->CreateProgramFromFile("kernels/convolution-db.cl");
        prog_mad = _ocl_base->CreateProgramFromFile("kernels/mad.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog, "grayscale_rgba");
        _ocl_base->CreateKernelFromProgram(prog, "resize");
        _ocl_base->CreateKernelFromProgram(prog_average, "average");
        _ocl_base->CreateKernelFromProgram(prog_ma, "matrix_addition");
        _ocl_base->CreateKernelFromProgram(prog_cv, "convolution_image");
        _ocl_base->CreateKernelFromProgram(prog_cv_d, "convolution_double");
        _ocl_base->CreateKernelFromProgram(prog_mad, "mad");
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
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *)&aBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *)&bBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *)&rBuffer);

        size_t global_work_size[2];
        global_work_size[0] = matwidth * sizeof(int);
        global_work_size[1] = matheight * sizeof(int);

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

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *)&aBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *)&bBufferd);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *)&rBufferd);

        size_t global_work_size[2];
        global_work_size[0] = matwidth * sizeof(float);
        global_work_size[1] = matheight * sizeof(float);

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

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

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

    unsigned convolution_double(int windowSize)
    {
        //Creating OpenCL buffers for matrices
        cl_mem inputImBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        28 * 28 * sizeof(float),
                                        inputIm,
                                        NULL);

        cl_mem weightsBuffer = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        avg_window_size * avg_window_size * sizeof(float),
                                        matrixWd,
                                        NULL);

        cl_mem outputLayerBuffer = clCreateBuffer(_ocl_base->context,
                                          CL_MEM_READ_WRITE,
                                          28 * 28 * sizeof(float),
                                          NULL,
                                          NULL);
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *)inputImBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *)outputLayerBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 2, sizeof(cl_mem), (void *)weightsBuffer);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 3, sizeof(int), &windowSize);

        size_t global_work_size[2];
        global_work_size[0] = 28 * sizeof(float);
        global_work_size[1] = 28 * sizeof(float);

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

        kernel_execution_times[5] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        //Reading result from GPU memory to main memory
        status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                     outputLayerBuffer,
                                     0,
                                     0,
                                     28 * 28 * sizeof(float),
                                     layer1,
                                     0,
                                     NULL,
                                     NULL);

        return (unsigned)status;
    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times\n\n";
        std::cout << "  Matrix addition: " << kernel_execution_times[5] << " us\n\n";

        std::cout << "  Image1:\n";
        img1->print_kernel_execution_times();
    }

    std::unique_ptr<OCL_image> img1;
    std::unique_ptr<OCL_Base> _ocl_base;

private:
    cl_program prog;
    cl_program prog_average;
    cl_program prog_ma;
    cl_program prog_cv;
    cl_program prog_cv_d;
    cl_program prog_mad;

    cl_event _event;

    // 0 - Grayscale
    // 1 - Resize
    // 2 - Average
    // 4 - Matrix addition
    // 5 - convolution
    unsigned long kernel_execution_times[6] = {0, 0, 0, 0, 0, 0};
};

OCL_Phase1 ocl_phase1;

    struct LoadImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.load_image("../../source-img/im0.png");
            return (int) error;
        }
    };
    
    struct TransformToGreyscale : public IProgram
    {
        int run() override
        {
            unsigned error = img0.transform_to_grayscale();
            return (int) error;
        }
    };

    struct ResizeImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.resize_image(scaling_factor);
            return (int) error;
        }
    };

    struct ApplyFilter : public IProgram
    {
        int run() override
        {
            unsigned char * img0_fi = (unsigned char*)malloc(img0.get_width() * img0.get_height());
            img0.clone_image(img0_fi);

            unsigned char * img0_fo = (unsigned char*)malloc(img0.get_width() * img0.get_height());
            img0.clone_image(img0_fo);

            movingAvgFilter(img0_fo, img0_fi, img0.get_width(), img0.get_height(), 5);

            img0.set_image(img0_fo, img0.get_width(), img0.get_height(), GREY_CHANNELS);

            free(img0_fi);
            free(img0_fo);

            return 0;
        }
    };

    struct SaveGreyscaleImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.save_image("../../output-img/im0_grey.png");
            return (int) error;
        }
    };

    struct SaveResizedImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.save_image("../../output-img/im0_grey_resized.png");
            return (int) error;
        }
    };

    struct SaveFilteredImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.save_image("../../output-img/im0_grey_average_filtered.png");
            return (int) error;
        }
    };

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

            unsigned error = 0;
            return (int) error;
        }
    };

    struct MatrixAdditionC : public IProgram
    {
        int run() override
        {

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    matrixRcpp[i * matwidth + j] = matrixA[i * matwidth + j] + matrixB[i * matwidth + j];
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

            unsigned error = 0;
            return (int) error;
        }
    };

    struct SaveMatrixCpp : public IProgram
    {
        int run() override
        {

            FILE *f = fopen("../../output-img/p1-matAddCpp.txt", "w");
            if (f == NULL)
            {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    fprintf(f, "%d ", matrixRcpp[i * matwidth + j]);
                }
                fprintf(f, "\n");
            }

            fclose(f);

            unsigned error = 0;
            return (int) error;
        }
    };

    struct SaveMatrixOCL : public IProgram
    {
        int run() override
        {

            FILE *fp = fopen("../../output-img/p1-matAddOcl.txt", "w");
            if (fp == NULL)
            {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    fprintf(fp, "%d ", matrixRocl[i * matwidth + j]);
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            fp = fopen("../../output-img/p1-madOcl.txt", "w");
            if (fp == NULL)
            {
                printf("Error opening file!\n");
                exit(1);
            }

            for (int i=0; i<matheight; i++) {
                for (int j=0; j<matwidth; j++) {
                    fprintf(fp, "%f ", matrixRocld[i * matwidth + j]);
                }
                fprintf(fp, "\n");
            }

            fclose(fp);

            unsigned error = 0;
            return (int) error;
        }
    };

    struct OclLoadImage : public IProgram
    {
        int run() override
        {
            unsigned error = ocl_phase1.img1->load_image("../../source-img/im1.png");
            return (int) error;
        }
    };

    struct OclTransformToGreyscale : public IProgram
    {
        int run() override
        {
            ocl_phase1.img1->transform_to_grayscale();
            return 0;
        }
    };

    struct OclSaveGreyscaleImage : public IProgram
    {
        int run() override
        {
            unsigned error = ocl_phase1.img1->save_image("../../output-img/im1_grey.png");
            return (int) error;
        }
    };

    struct OclResizeImage : public IProgram
    {
        int run() override
        {
            ocl_phase1.img1->resize_image(scaling_factor);
            return 0;
        }
    };

    struct OclSaveResizedImage : public IProgram
    {
        int run() override
        {
            unsigned error = ocl_phase1.img1->save_image("../../output-img/im1_grey_resized.png");
            return (int) error;
        }
    };
    
    struct OclAverage : public IProgram
    {
        int run() override
        {
            ocl_phase1.img1->average(avg_window_size);
            return 0;
        }
    };

    struct OclSaveFilteredImage : public IProgram
    {
        int run() override
        {
            ocl_phase1.img1->save_image("../../output-img/im1_grey_average_filtered.png");
            return 0;
        }
    };

    struct CreateWeights : public IProgram
    {
        int run() override
        {
            matrixW = (int*)malloc((avg_window_size * avg_window_size) * sizeof(int));
            matrixWd = (float*)malloc((avg_window_size * avg_window_size) * sizeof(float));
            inputIm = (float*)malloc((28 * 28) * sizeof(float));
            layer1 = (float*)malloc((28 * 28) * sizeof(float));

            for (int i=0; i<avg_window_size; i++) {
                for (int j=0; j<avg_window_size; j++) {
                    matrixW[i * avg_window_size + j] = i + j + 1;
                    matrixWd[i * avg_window_size + j] = 2;
                }
            }

            for (int i=0; i<28; i++) {
                for (int j=0; j<28; j++) {
                    inputIm[i * 28 + j] = 1;
                }
            }

            for (int i=0; i<28; i++) {
                for (int j=0; j<28; j++) {
                    layer1[i * 28 + j] = 1;
                }
            }

            return 0;
        }
    };

struct SaveLayer : public IProgram
{
    int run() override
    {

        FILE *fp = fopen("../../output-img/p1-layer1.txt", "w");
        if (fp == NULL)
        {
            printf("Error opening file!\n");
            exit(1);
        }

        for (int i=0; i<28; i++) {
            for (int j=0; j<28; j++) {
                fprintf(fp, "%f ", layer1[i * 28 + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);

        unsigned error = 0;
        return (int) error;
    }
};

    struct Convolution : public IProgram
    {
        int run() override
        {
            ocl_phase1.img1->convolution(avg_window_size);
            ocl_phase1.img1->save_image("../../output-img/im1_convolution.png");

            ocl_phase1.convolution_double(5);
            return 0;
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
    LoadImage loadImage;
    TransformToGreyscale transformToGreyscale;
    ResizeImage resizeImage;
    ApplyFilter applyFilter;
    SaveGreyscaleImage saveGreyscaleImage;
    SaveResizedImage saveResizedImage;
    SaveFilteredImage saveFilteredImage;

    //Step 2
    CreateMatrices createMatrices;
    MatrixAdditionC matrixAdditionC;
    MatrixAdditionOCL matrixAdditionOCL;
    SaveMatrixCpp saveMatrixCpp;
    SaveMatrixOCL saveMatrixOCL;

    //Step 3
    OclLoadImage oclLoadImage;
    OclTransformToGreyscale oclTransformToGreyscale;
    OclSaveGreyscaleImage oclSaveGreyscaleImage;
    OclResizeImage oclResizeImage;
    OclSaveResizedImage oclSaveResizedImage;
    OclAverage oclAverage;
    OclSaveFilteredImage oclSaveFilteredImage;

    //Step 4
    Convolution convolution;
    CreateWeights createWeights;
    SaveLayer saveLayer;

    int result = Program_sw.runProgram(loadImage);
    //Step 1
    std::cout << "Step 1" << std::endl;

    std::cout << "Load image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(transformToGreyscale);
    std::cout << "Transform to greyscale return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveGreyscaleImage);
    std::cout << "Save greyscale image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(resizeImage);
    std::cout << "Resize greyscale image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveResizedImage);
    std::cout << "Save resized image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(applyFilter);
    std::cout << "Apply filter return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveFilteredImage);
    std::cout << "Save filtered image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    //Step 2
    std::cout << "Step 2" << std::endl;

    result = Program_sw.runProgram(createMatrices);
    std::cout << "Creation of matrices: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(matrixAdditionC);
    std::cout << "C++ matrix addition: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveMatrixCpp);
    std::cout << "C++ matrix save: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(matrixAdditionOCL);
    std::cout << "OpenCL matrix addition: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveMatrixOCL);
    std::cout << "OpenCL matrix save: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    //Step 3
    std::cout << "Step 3" << std::endl;

    result = Program_sw.runProgram(oclLoadImage);
    std::cout << "OpenCl: Load image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclTransformToGreyscale);
    std::cout << "OpenCl: Transform to greyscale return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclSaveGreyscaleImage);
    std::cout << "OpenCl: Save greyscale image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclResizeImage);
    std::cout << "OpenCl: Resize image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclSaveResizedImage);
    std::cout << "OpenCl: Save resized image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclAverage);
    std::cout << "OpenCl: Calculate average result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(oclSaveFilteredImage);
    std::cout << "OpenCl: Save image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    //Step 4
    std::cout << "Step 4 - convolution" << std::endl;
    result = Program_sw.runProgram(createWeights);
    result = Program_sw.runProgram(convolution);
    result = Program_sw.runProgram(saveLayer);
    std::cout << "OpenCl: Save image result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl << std::endl;

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl_phase1.print_kernel_execution_times();

    free(matrixA);
    free(matrixB);
    free(matrixRcpp);
    free(matrixRocl);
    free(matrixW);
    free(matrixWd);

    printPlatformProfile(false);
    return 0;
}
