__kernel void average(__global unsigned char *image, __global float *out, int width, int height,
                      int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);

  col = col * 4;

  int windowSizeHalf = windowSize / 2;
  float sum = 0;
  int count = 0;

  float sum3 = 0;
  int count3 = 0;

  float sum4 = 0;
  int count4 = 0;

  float sum5 = 0;
  int count5 = 0;


  // Loop through the window to get the average value
  for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
    for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
      int x2 = col + j;
      int x3 = col + j + 1;
      int x4 = col + j + 2;
      int x5 = col + j + 3;

      int y2 = row + i;

      // Check that the pixel is inside the image
      if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum += image[y2 * width + x2];
        count++;
      }

      // Check that the pixel is inside the image
      if (x3 >= 0 && x3 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum3 += image[y2 * width + x3];
        count3++;
      }

      // Check that the pixel is inside the image
      if (x4 >= 0 && x4 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum4 += image[y2 * width + x4];
        count4++;
      }

      // Check that the pixel is inside the image
      if (x5 >= 0 && x5 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum5 += image[y2 * width + x5];
        count5++;
      }

    }
  }

  if (count == 0) {
    count = 1;
  }

  if (count3 == 0) {
    count3 = 1;
  }

  if (count4 == 0) {
    count4 = 1;
  }

  if (count5 == 0) {
    count5 = 1;
  }


  float avg = sum / count;
  float avg3 = sum3 / count3;
  float avg4 = sum4 / count4;
  float avg5 = sum5 / count5;

  // Check that the pixel is inside the image
    out[row * width + col] = avg;
    out[row * width + col + 1] = avg3;
    out[row * width + col + 2] = avg4;
  if ((col + 3) < width) {
    out[row * width + col + 3] = avg5;
  }

}
