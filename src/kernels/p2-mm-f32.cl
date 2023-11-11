__kernel void mad(__global float* inputA, __global float* inputB, __global float* output) {
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
