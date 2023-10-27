#ifndef UTIL_H
#define UTIL_H
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/opencl.h>

int printPlatformProfile(bool print_extras);

void movingAvgFilter(unsigned char* imageOut, unsigned char *image, unsigned width, unsigned height, unsigned windowSize);
void ZNCCFilter(unsigned char* imageOut, unsigned char* image, unsigned char* image2, unsigned width, unsigned height, unsigned windowSize, unsigned leftToRight);
void ZNCCFilterOptimizedC(unsigned char* imageOut, unsigned char* imageOut2, unsigned char* image, unsigned char* image2, unsigned width, unsigned height, unsigned windowSize);
void OMP_ZNCCFilterOptimizedC(unsigned char* imageOut, unsigned char* imageOut2, unsigned char* image, unsigned char* image2, unsigned width, unsigned height, unsigned windowSize);

bool differenceIsOverThreshold(unsigned char pixel1, unsigned char pixel2, int threshold);

unsigned char crossCheck(unsigned char pixel1, unsigned char pixel2, int threshold);
void crossCheckTwoImages(unsigned char *image1, unsigned char *image2, int threshold, unsigned char *outputImage, int imageSize);

#endif // UTIL_H