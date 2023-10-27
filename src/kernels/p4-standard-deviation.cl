__kernel void standardDeviation(__global unsigned char *image, __global float *out,
                                __global float *avg, int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  int windowSizeHalf = windowSize / 2;
  float sum = 0;

  for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
    for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
      int x2 = col + j;
      int y2 = row + i;

      // Check that the pixel is inside the image
      if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum += pow((image[y2 * width + x2] - avg[row * width + col]), 2);
      }
    }
  }

  float std = pow(sum, (float)0.5);

  out[row * width + col] = std;
}