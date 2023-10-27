__kernel void convolution_double(__global float *layer, __global float *output, __global float *weight,
                      int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  output[row * width + col] = 4.0;
}
