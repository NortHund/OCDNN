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

__kernel void mm_int_sumrow(__global int* inputA, __global int* inputB, __global int* output, __global int* sumRow) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    for (int i = 0; i < width; i++) {
        sumRow[i] = 0;
        for (int j = 0; j < height; j++) {
            sumRow[i] += inputA[(j * width) + i];
        }
    }
}

__kernel void mm_int_sumcol(__global int* inputA, __global int* inputB, __global int* output, __global int* sumCol) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    sumCol[row] = 0;
    for (int j = 0; j < width; j++) {
        sumCol[row] += inputB[(row * width) + j];
    }

}

__kernel void mm_int_ics(__global int* inputA, __global int* inputB, __global int* ics) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int checksum = 0;

    for (int i = 0; i < width; i++) {
        checksum += inputA[i] * inputB[i];
    }

    ics[0] = checksum;

}



