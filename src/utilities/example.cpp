#include "utils.h"

int main()
{
    // Create one dimensional image of 3x3 pixels
    unsigned char image[9] = { 1, 1, 1, 1, 10, 1, 1, 1, 1 };
    unsigned char outImage[9]; 

    // take moving average of image
    movingAvgFilter(outImage, image, 3, 3, 3);

    // print out image as matrix
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << (int)outImage[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }

    printPlatformProfile(true);
    return 0;
}
