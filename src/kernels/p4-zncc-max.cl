__kernel void znccMax(__global unsigned char *img1, __global unsigned char *img2,
                      __global float *avg1, __global float *avg2,
                      __global float *std1, __global float *std2,
                      __global unsigned char *DispMaxOut, int windowSize,
                      int leftToRight, int maxDisp) {

  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);

  int windowSizeHalf = windowSize / 2;
  int sum = 0;
  int x2r = 0;
  int x2mr = 0;
  int minDisp = 0;
  int dispRange = maxDisp - minDisp;
  float zncc = 0;
  float znccVal = 0;

  for (int d = minDisp; d < maxDisp; d++) {
    sum = 0;

    // Loop through the window to get the ZNCC value
    for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
      for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
        int x2 = col + j;
        int y2 = row + i;

        if (leftToRight == 1) {
          x2r = x2 - d;
          x2mr = col - d;
        }
        else {
          x2r = x2 + d;
          x2mr = col + d;
        }

        // Check that the pixel is inside the image
        if (x2 >= 0 && x2 < (int)width 
            && x2r >= 0 && x2r < (int)width 
            && y2 >= 0 && y2 < (int)height 
            && x2mr >= 0 && x2mr < (int)width) 
            {
          sum += (img1[y2 * width + x2] - avg1[row * width + col]) *
                 (img2[y2 * width + x2r] - avg2[row * width + x2mr]);
        }
      }
    }

    if (leftToRight == 1) {
      x2mr = col - d;
    }
    else {
      x2mr = col + d;
    }

    if (x2mr >= 0 && x2mr < (int)width) {
      znccVal = sum / (std1[row * width + col] * std2[row * width + x2mr]);
    }

    if (znccVal > zncc) {
      zncc = znccVal;
      DispMaxOut[row * width + col] = (255 * abs(d)) / dispRange;
    }
  }
}
