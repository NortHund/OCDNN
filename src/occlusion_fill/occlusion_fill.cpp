#include "occlusion_fill.h"


int getNearestFillPixelValue(int pixel_index, int pix_x, int pix_y, unsigned char *image, int width, int height)
{
    int fillPixelValue = 0;
    int pixelValue = 0;
    int windowSizeHalf = 34;

    int minDist = width + height;

    // Loop through the window to get the average value
    for (int i = -windowSizeHalf; i <= windowSizeHalf; i++) {
        for (int j = -windowSizeHalf; j <= windowSizeHalf; j++) {
            int x2 = pix_x + j;
            int y2 = pix_y + i;

            // Check that the pixel is inside the image
            if (x2 >= 0 && x2 < (int)width &&  y2 >= 0 && y2 < (int)height) {
                int dist = euclideanDistanceBetweenTwoPixels(pixel_index, x2, y2, width);
                pixelValue = getPixelValueFromImage(x2, y2, image, width);

                if (!pixelIsZero(pixelValue)
                    && pixelDistanceIsLessThanLastMinimum(dist, minDist))
                {
                    minDist = dist;
                    fillPixelValue = pixelValue;
                }
                
            }
        }
    }
     
    return fillPixelValue;
}

void fillZeroPixels(unsigned char *image, unsigned char *outImage, int width, int height)
{
    int pixel_index = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (image[pixel_index] == 0)
            {
                outImage[pixel_index] = getNearestFillPixelValue(pixel_index, x, y, image, width, height);
            }
            else
            {
                outImage[pixel_index] = image[pixel_index];
            }
            pixel_index++;
        }
    }
}

int pixelRow(int pixel_index, int width)
{
    return pixel_index / width;
}

int verDistance(int pixel_index, int row_index, int width)
{
    return abs(pixelRow(pixel_index, width) - row_index);
}

int pixelColumn(int pixel_index, int width)
{
    return pixel_index % width;
}

int horDistance(int pixel_index, int column_index, int width)
{
    return abs(pixelColumn(pixel_index, width) - column_index);
}

int euclideanDistanceBetweenTwoPixels(int pixel1_index, int pixel2_x_index, int pixel2_y_index, int image_width)
{
    return sqrt(pow(horDistance(pixel1_index, pixel2_x_index, image_width), 2) + pow(verDistance(pixel1_index, pixel2_y_index, image_width), 2));
}

int getPixelValueFromImage(int pixel_x_index, int pixel_y_index, unsigned char *image, int width)
{
    return *(image + pixel_x_index + pixel_y_index * width);
}

bool pixelIsZero(int pixel_value)
{
    return pixel_value == 0;
}

bool pixelDistanceIsLessThanLastMinimum(int distance, int last_min_distance)
{
    return distance < last_min_distance;
}
