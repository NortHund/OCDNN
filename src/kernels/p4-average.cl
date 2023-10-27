__kernel void average(__global unsigned char *image, __global float *out,
                      int windowSize) {
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  int windowSizeHalf = windowSize / 2;
  float sum = 0;
  int count = 0;

  // Loop through the window to get the average value
  for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
    for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
      int x2 = col + j;
      int y2 = row + i;

      // Check that the pixel is inside the image
      if (x2 >= 0 && x2 < (int)width && y2 >= 0 && y2 < (int)height) {
        sum += image[y2 * width + x2];

        count++;
      }
    }
  }

  if (count == 0) {
    count = 1;
  }

  float avg = sum / count;

  out[row * width + col] = avg;
}
