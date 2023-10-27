#include "lodepng_wrapper.h"

int main()
{
    lodepng_wrapper::LodepngWrapper img0;
    unsigned error = img0.load_image("../../../source-img/im0.png");
    if (error) { return (int) error; }
    error = img0.save_image("../../../output-img/im0.png");
    error = img0.transform_to_grayscale();
    error = img0.save_image("../../../output-img/im0_grey.png");
    return (int) error;
}
