__kernel void convolution_double(__global double* inputA, __global double* inputB, __global double* output, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = output[(layerB * width * height) + (row * width) + col];

    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
          sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)] * inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
          //sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)];
          //sum += inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
        }
    }

  output[(layerB * width * height) + (row * width) + col] = sum;
}

__kernel void input_sum(__global double* input, __global double* output, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

    for (int i = 0; i < depth; i++) {
      sum += input[(i * height * width) + (row * width) + (col)];
    }

  output[(row * width) + col] = sum;
}

__kernel void output_sum(__global double* input, __global double* output, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

    for (int i = 0; i < depth; i++) {
      sum += input[(i * height * width) + (row * width) + (col)];
    }

  output[(row * width) + col] = sum;
}

__kernel void cs_compare(__global double* inputCs, __global double* outputCs, __global double* result, int layer) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    int wgNum = get_group_id(0);

    double diff = 0;

    diff = fabs(inputCs[(row * width) + col] - outputCs[(row * width) + col]);

    //change this to a very low value and some results will start failing
    if (diff > 0.0000000000001) {
        result[layer] = diff + 1;
    }
    //result[layer] = diff + 1;



}