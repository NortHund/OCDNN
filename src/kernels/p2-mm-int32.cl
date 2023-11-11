__kernel void mm_int(__global int* inputA, __global int* inputB, __global int* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

__kernel void mm_short(__global short* inputA, __global short* inputB, __global short* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    short out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

__kernel void mm_char(__global char* inputA, __global char* inputB, __global char* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    char out = 0;

    for(int i= 0; i < width; i++) {
        out += inputA[row * width + i] * inputB[i * width + col];
    }

    output[row * width + col] = out;
}

