#ifndef OCCLUSION_FILL_H
#define OCCLUSION_FILL_H

#include <math.h>

void fillZeroPixels(unsigned char *image, unsigned char *outImage, int width, int height);
int getNearestFillPixelValue(int pixel_index, int pix_x, int pix_y, unsigned char *image, int width, int height);

int pixelRow(int pixel_index, int width);
int verDistance(int pixel_index, int row_index, int width);
int pixelColumn(int pixel_index, int width);
int horDistance(int pixel_index, int column_index, int width);
int euclideanDistanceBetweenTwoPixels(int pixel1_index, int pixel2_x_index, int pixel2_y_index, int image_width);
int getPixelValueFromImage(int pixel_x_index, int pixel_y_index, unsigned char *image, int width);
bool pixelIsZero(int pixel_value);
bool pixelDistanceIsLessThanLastMinimum(int distance, int last_min_distance);

#endif  // OCCLUSION_FILL_H