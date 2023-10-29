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
