__kernel void mm_double(__global double* inputA, __global double* inputB, __global double* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

__kernel void mm_float(__global float* inputA, __global float* inputB, __global float* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    float out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

__kernel void mm_half(__global half* inputA, __global half* inputB, __global half* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    half out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

