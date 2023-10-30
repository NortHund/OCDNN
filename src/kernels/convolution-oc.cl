__kernel void convolution_oc(__global float* inputA, __global float* checksum, int heightA, int widthA, int depthA) {
    int id = get_global_id(0);
    float sum = 0;

    for (int m = 0; m < depthA; m++) {
        for (int i = 0; i < heightA; i++) {
            for (int j = 0; j < widthA; j++) {
              sum += inputA[(m * heightA * widthA) + (i * widthA) + j];
            }
        }
    }

    checksum[id] = sum;
}