unsigned char crossCheck(unsigned char pixel1,unsigned char pixel2, int threshold);
bool differenceIsOverThreshold(unsigned char pixel1,unsigned char pixel2, int threshold);

__kernel void cross_check(__global unsigned char* image1, __global unsigned char* image2, int threshold, __global unsigned char* outImage)
{
  int col = get_global_id(0);
  int row = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);
  int pixel_index = (row*width + col);

  outImage[pixel_index] = crossCheck(image1[pixel_index], image2[pixel_index], threshold);

}

//cross check functions
bool differenceIsOverThreshold(unsigned char pixel1, unsigned char pixel2, int threshold)
{
    return abs(pixel1 - pixel2) > threshold;
}

unsigned char crossCheck(unsigned char pixel1, unsigned char pixel2, int threshold)
{
    unsigned char returnValue = 0;
    if (!differenceIsOverThreshold(pixel1, pixel2, threshold))
    {
        returnValue = pixel1;
    }

    return returnValue;
}
