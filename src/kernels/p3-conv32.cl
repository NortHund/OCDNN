__kernel void convolution_double(__global double* inputA, __global double* inputB, __global double* output, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
          sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)] * inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
        }
    }

  output[(layerB * width * height) + (row * width) + col] = sum;
}

__kernel void convolution_fl(__global float* inputA, __global float* inputB, __global float* output, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    float sum = 0;

    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
          sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)] * inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
        }
    }

  output[(layerB * width * height) + (row * width) + col] = sum;
}

__kernel void convolution_double_ics(__global double* inputA, __global double* inputB, __global double* ics, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double wSum = 0;
    double xSum = 0;
    double checksum = 0;
    for (int n = 0; n < depthA; ++n) {
        for (int i = 0; i < heightB; ++i) {
            for (int j = 0; j < widthB; ++j) {
                wSum = 0;
                for (int m = 0; m < depthB; ++m) {
                    wSum += inputB[(n * depthB * heightB * widthB) + (m * heightB * widthB) + (i * widthB) + j];
                }
                //printf("wSum: %f\n", wSum);
                xSum = 0;
                for (int r = 0; r < height; ++r) {
                    for (int c = 0; c < width; ++c) {
                        xSum += inputA[(n * heightA * widthA) + ((r + i) * widthA) + (c + j)];
                    }
                }
                //printf("xSum: %f\n", xSum);
                checksum += xSum * wSum;
                //printf("n: %d, i: %d, j: %d\n checksum: %f, xSum: %f, wSum: %f\n", n, i, j, checksum, xSum, wSum);
            }
        }
    }

  ics[0] = checksum;
}

__kernel void convolution_double_ocs(__global double* inputA, __global double* ocs, int heightA, int widthA, int depthA, int ind) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

        for (int m = 0; m < depthA; m++) {
            for (int i = 0; i < heightA; i++) {
                for (int j = 0; j < widthA; j++) {
                  sum += inputA[(m * heightA * widthA) + (i * widthA) + j];
                }
            }
        }

  ocs[ind] = sum;
}