__kernel void standardDeviation(__global unsigned char *image, __global float *out, 
                                __global float *avg,
                                int width, int height, int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);

  col = col * 4;
  int imSize = width * height;

  int windowSizeHalf = windowSize / 2;
  float sum = 0;
  float sum2 = 0;
  float sum3 = 0;
  float sum4 = 0;

  for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
    for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
      int x2 = col + j;
      int x3 = col + j + 1;
      int x4 = col + j + 2;
      int x5 = col + j + 3;

      int y2 = row + i;

      // Check that the pixel is inside the image
      if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum += pow((image[y2 * width + x2] - avg[row * width + col]), 2);
      }
      if (x3 >= 0 && x3 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum2 += pow((image[y2 * width + x3] - avg[row * width + col + 1]), 2);
      }
      if (x4 >= 0 && x4 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum3 += pow((image[y2 * width + x4] - avg[row * width + col + 2]), 2);
      }
      if (x5 >= 0 && x5 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum4 += pow((image[y2 * width + x5] - avg[row * width + col + 3]), 2);
      }

    }
  }

  float std = pow(sum, (float)0.5);
  float std2 = pow(sum2, (float)0.5);
  float std3 = pow(sum3, (float)0.5);
  float std4 = pow(sum4, (float)0.5);

    out[row * width + col] = std;
  if ((col + 1) < width) {
    out[row * width + col + 1] = std2;
  }
  if ((col + 2) < width) {
    out[row * width + col + 2] = std3;
  }
  if ((col + 3) < width) {
    out[row * width + col + 3] = std4;
  }
  

}