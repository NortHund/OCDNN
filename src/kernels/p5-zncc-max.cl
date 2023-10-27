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
  int sum2 = 0;
  int sum3 = 0;
  int sum4 = 0;
  int sum5 = 0;

  int x2r = 0;
  int x3r = 0;
  int x4r = 0;
  int x5r = 0;
  int x6r = 0;

  int x2mr = 0;
  int x3mr = 0;
  int x4mr = 0;
  int x5mr = 0;
  int x6mr = 0;

  int minDisp = 0;
  int dispRange = maxDisp - minDisp;

  float zncc = 0;
  float znccVal = 0;
  float znccVal2 = 0;
  float znccVal3 = 0;
  float znccVal4 = 0;
  float znccVal5 = 0;

  for (int d = minDisp; d < maxDisp; (d += 5)) {
    sum = 0;
    sum2 = 0;
    sum3 = 0;
    sum4 = 0;
    sum5 = 0;

    znccVal = 0;
    znccVal2 = 0;
    znccVal3 = 0;
    znccVal4 = 0;
    znccVal5 = 0;

    // Loop through the window to get the ZNCC value
    for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
      for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
        int x2 = col + j;
        int y2 = row + i;

        if (leftToRight == 1) {
          x2r = x2 - d;
          x3r = x2 - (d + 1);
          x4r = x2 - (d + 2);
          x5r = x2 - (d + 3);
          x6r = x2 - (d + 4);

          x2mr = col - (d);
          x3mr = col - (d + 1);
          x4mr = col - (d + 2);
          x5mr = col - (d + 3);
          x6mr = col - (d + 4);
        }
        else {
          x2r = x2 + d;
          x3r = x2 + (d + 1);
          x4r = x2 + (d + 2);
          x5r = x2 + (d + 3);
          x6r = x2 + (d + 4);

          x2mr = col + (d);
          x3mr = col + (d + 1);
          x4mr = col + (d + 2);
          x5mr = col + (d + 3);
          x6mr = col + (d + 4);
        }

        // Check that the pixel is inside the image
        if (y2 >= 0 && y2 < (int)height && x2 >= 0 && x2 < (int)width) {

          if (x2r >= 0 && x2r < (int)width 
              && x2mr >= 0 && x2mr < (int)width) 
          {
            sum += (img1[y2 * width + x2] - avg1[row * width + col]) *
                  (img2[y2 * width + x2r] - avg2[row * width + x2mr]);
          }
          if (x3r >= 0 && x3r < (int)width
              && x3mr >= 0 && x3mr < (int)width) 
          {
            sum2 += (img1[y2 * width + x2] - avg1[row * width + col]) *
                  (img2[y2 * width + x3r] - avg2[row * width + x3mr]);
          }
          if (x4r >= 0 && x4r < (int)width
              && x4mr >= 0 && x4mr < (int)width) 
          {
            sum3 += (img1[y2 * width + x2] - avg1[row * width + col]) *
                  (img2[y2 * width + x4r] - avg2[row * width + x4mr]);
          }
          if (x5r >= 0 && x5r < (int)width
              && x5mr >= 0 && x5mr < (int)width) 
          {
            sum4 += (img1[y2 * width + x2] - avg1[row * width + col]) *
                  (img2[y2 * width + x5r] - avg2[row * width + x5mr]);
          }
          if (x6r >= 0 && x6r < (int)width
              && x6mr >= 0 && x6mr < (int)width) 
          {
            sum5 += (img1[y2 * width + x2] - avg1[row * width + col]) *
                  (img2[y2 * width + x6r] - avg2[row * width + x6mr]);
          }

        }

      }
    }

    if (leftToRight == 1) {
      x2mr = col - (d);
      x3mr = col - (d + 1);
      x4mr = col - (d + 2);
      x5mr = col - (d + 3);
      x6mr = col - (d + 4);
    }
    else {
      x2mr = col + (d);
      x3mr = col + (d + 1);
      x4mr = col + (d + 2);
      x5mr = col + (d + 3);
      x6mr = col + (d + 4);
    }

    if (x2mr >= 0 && x2mr < (int)width) {
      znccVal = sum / (std1[row * width + col] * std2[row * width + x2mr]);
    }
    if (x3mr >= 0 && x3mr < (int)width) {
      znccVal2 = sum2 / (std1[row * width + col] * std2[row * width + x3mr]);
    }
    if (x4mr >= 0 && x4mr < (int)width) {
      znccVal3 = sum3 / (std1[row * width + col] * std2[row * width + x4mr]);
    }
    if (x5mr >= 0 && x5mr < (int)width) {
      znccVal4 = sum4 / (std1[row * width + col] * std2[row * width + x5mr]);
    }
    if (x6mr >= 0 && x6mr < (int)width) {
      znccVal5 = sum5 / (std1[row * width + col] * std2[row * width + x6mr]);
    }

    if (znccVal > zncc) {
      zncc = znccVal;
      DispMaxOut[row * width + col] = (255 * abs(d)) / dispRange;
    }
    if (znccVal2 > zncc) {
      zncc = znccVal2;
      DispMaxOut[row * width + col] = (255 * abs(d+1)) / dispRange;
    }
    if (znccVal3 > zncc) {
      zncc = znccVal3;
      DispMaxOut[row * width + col] = (255 * abs(d+2)) / dispRange;
    }
    if (znccVal4 > zncc) {
      zncc = znccVal4;
      DispMaxOut[row * width + col] = (255 * abs(d+3)) / dispRange;
    }
    if (znccVal5 > zncc) {
      zncc = znccVal5;
      DispMaxOut[row * width + col] = (255 * abs(d+4)) / dispRange;
    }

  }

  //DispMaxOut[row * width + col] = avg1[row * width + col];
  //DispMaxOut[row * width + col] = std1[row * width + col];
}
