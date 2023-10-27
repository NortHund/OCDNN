__kernel void convolution(__global unsigned char *layer, __global unsigned char *output, __global unsigned char *weight,
                      int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  float sum = 0;
  int count = 0;
  
  for (int i = 0; i < windowSize; i++) {
    for (int j = 0; i < windowSize; i++) {
      //sum += layer[(row + i) * width + (col + j)];
      int x2 = col + j;
      int y2 = row + i;

      // Check that the pixel is inside the image
      if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum += layer[y2 * width + x2] * weight[i * windowSize + j];
      }
    }
  }

  output[row * width + col] = sum / 8;
}
