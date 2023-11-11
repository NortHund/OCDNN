#include "opencl_base.h"

OCL_Base::OCL_Base()
{
    platform = NULL;
    status = 0;

    numDevices = 0;

    Programs = NULL;
    ProgCount = 0;

    Kernels = NULL;
    KernCount = 0;

    Init();
}

OCL_Base::~OCL_Base()
{
    status = clReleaseCommandQueue(commandQueue); //Release  Command queue.
    status = clReleaseContext(context); //Release context.

    if (devices != NULL)
    {
        free(devices);
        devices = NULL;
    }

    if (Programs != NULL)
    {
        free(Programs);
        Programs = NULL;
    }

    if (Kernels != NULL)
    {
        free(Kernels);
        Kernels = NULL;
    }
}

cl_program OCL_Base::CreateProgramFromFile(const char* filename)
{
    cl_program program;

    // Save programs to list and keep count how many programs are created
    if (Programs == NULL)
    {
        Programs = (cl_program*)malloc(sizeof(cl_program));
    }
    else
    {
        Programs = (cl_program*)realloc(Programs, sizeof(cl_program) * (ProgCount + 1));
    }

    /*Step 5: Create program object */
    std::string sourceStr;
    status = convertToString(filename, sourceStr);
    const char *source = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source) };
    program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);

    /*Step 6: Build program. */
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    Programs[ProgCount] = program;
    ProgCount++;

    return program;
}

cl_kernel OCL_Base::CreateKernelFromProgram(cl_program program, const char* kernelName)
{
    cl_kernel kernel;

    // Save kernels to list and keep count how many kernels are created
    if (Kernels == NULL)
    {
        Kernels = (cl_kernel*)malloc(sizeof(cl_kernel));
    }
    else
    {
        Kernels = (cl_kernel*)realloc(Kernels, sizeof(cl_kernel) * (KernCount + 1));
    }

    kernel = clCreateKernel(program, kernelName, NULL);

    Kernels[KernCount] = kernel;
    KernCount++;

    return kernel;
}

void OCL_Base::Init()
{
    /*Step1: Getting platforms and choose an available one.*/
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        std::cout << "Error: Getting platforms!" << std::endl;
    }

    /*For clarity, choose the first available platform. */
    if (numPlatforms > 0)
    {
        platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
        status = clGetPlatformIDs(numPlatforms, platforms, NULL);
        platform = platforms[0];
        free(platforms);
    }

    /*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if (numDevices == 0) //no GPU available.
    {
        std::cout << "No GPU device available." << std::endl;
        std::cout << "Choose CPU as default device." << std::endl;
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
    }
    else
    {
        devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    }

    /*Step 3: Create context.*/
    context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

    /*Step 4: Creating command queue associate with the context.*/
    commandQueue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, NULL);

}

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char*  str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));

    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return 0;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    std::cout << "Error: failed to open file\n:" << filename << std::endl;
    return 1;
}

int printPlatformInfo(bool print_extras)
{
    // Get the number of platforms
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms <= 0) {
        std::cout << "Failed to find any OpenCL platforms." << std::endl;
        return -1;
    }

    std::cout << "Platform count: " << num_platforms << std::endl;

    // Get the platform IDs
    cl_platform_id platforms[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);

    // Loop through the platforms
    for (cl_uint i = 0; i < num_platforms; i++) {
        // Print platform info
        char platform_version[100];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(platform_version), platform_version, NULL);
        std::cout << "Platform version: " << platform_version << std::endl;

        // Get the number of devices for this platform
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices <= 0) {
            std::cout << "Failed to find any OpenCL devices." << std::endl;
            return -1;
        }

        std::cout << "Device count on platform " << i << ": " << num_devices << std::endl;

        // Get the device IDs
        cl_device_id devices[num_devices];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

        // Loop through the devices
        for (cl_uint j = 0; j < num_devices; j++) {
            // Print device info
            char device_name[100];
            char device_version[100];
            char driver_version[100];
            char opencl_c_version[100];
            cl_uint device_max_compute_units; // number of parallel compute units
            cl_uint device_max_work_item_dimensions; // number of dimensions
            cl_device_local_mem_type local_mem_type; // CL_LOCAL or CL_GLOBAL
            cl_ulong local_mem_size; // in bytes
            cl_uint device_max_clock_frequency; // in MHz
            cl_ulong device_max_constant_buffer_size; // in bytes
            size_t device_max_work_group_size; // number of work-items
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);
            clGetDeviceInfo(devices[j], CL_DRIVER_VERSION, sizeof(driver_version), driver_version, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_OPENCL_C_VERSION, sizeof(opencl_c_version), opencl_c_version, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(device_max_compute_units), &device_max_compute_units, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(device_max_work_item_dimensions), &device_max_work_item_dimensions, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(device_max_clock_frequency), &device_max_clock_frequency, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(device_max_constant_buffer_size), &device_max_constant_buffer_size, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(device_max_work_group_size), &device_max_work_group_size, NULL);
            clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_TYPE, sizeof(local_mem_type), &local_mem_type, NULL);
            size_t device_max_work_item_sizes[device_max_work_item_dimensions];
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(device_max_work_item_sizes), &device_max_work_item_sizes, NULL);
            std::cout << "Device name: " << device_name << std::endl;
            std::cout << "Hardware version: " << device_version << std::endl;
            std::cout << "Driver version: " << driver_version << std::endl;
            std::cout << "OpenCL C version: " << opencl_c_version << std::endl;
            std::cout << "Parallel compute units: " << device_max_compute_units << std::endl;
            std::cout << "Device max work item dimensions: " << device_max_work_item_dimensions << std::endl;

            if (print_extras) {
                if (local_mem_type == CL_LOCAL) {
                    std::cout << "Device local mem type: CL_LOCAL" << std::endl;
                }
                else if (local_mem_type == CL_GLOBAL) {
                    std::cout << "Device local mem type: CL_GLOBAL" << std::endl;
                }
                else {
                    std::cout << "Device local mem type: UNKNOWN" << std::endl;
                }
                std::cout << "Device local mem size: " << local_mem_size << std::endl;
                std::cout << "Device max clock frequency: " << device_max_clock_frequency << std::endl;
                std::cout << "Device max constant buffer size: " << device_max_constant_buffer_size << std::endl;
                std::cout << "Device max work group size: " << device_max_work_group_size << std::endl;
                std::cout << "Device max work item sizes: ";
                for (cl_uint k = 0; k < device_max_work_item_dimensions; k++) {
                    std::cout << device_max_work_item_sizes[k];
                    if (k < device_max_work_item_dimensions - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    return 0;
}
