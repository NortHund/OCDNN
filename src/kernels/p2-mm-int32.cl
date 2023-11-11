__kernel void matrix_addition(__global int* inputA, __global int* inputB, __global int* output) {
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
