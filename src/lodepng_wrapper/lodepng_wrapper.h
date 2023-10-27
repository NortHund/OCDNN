// Wrapper created based on the example from the lodepng library
#ifndef LODEPNG_WRAPPER_HPP
#define LODEPNG_WRAPPER_HPP

#define RGBA_CHANNELS 4
#define GREY_CHANNELS 1

#include "lodepng.h"
#include <iostream>

namespace lodepng_wrapper
{

class LodepngWrapper
{
public:
    LodepngWrapper();
    ~LodepngWrapper();

    unsigned load_image(const char* filename);
    virtual unsigned save_image(const char* filename);
    virtual unsigned transform_to_grayscale();
    virtual void clone_image(unsigned char* dest);
    void set_image(unsigned char* src, unsigned width, unsigned height, int channels);
    void apply_filter(void (*filter)(unsigned char* image, unsigned width, unsigned height, unsigned windowSize), unsigned windowSize);
    void occlusion_fill(void (*filter)(unsigned char *image, unsigned char *outImage, int width, int height));
    virtual unsigned resize_image(unsigned scalingFactor);
    unsigned get_width();
    unsigned get_height();

protected:
    unsigned char* _image = 0;

    unsigned _width, _height;
    bool _imageIsGrayscaled;

private:
    unsigned save_RGBAimage(const char* filename);
    unsigned save_greyimage(const char* filename);
};

} // namespace lodepng_wrapper

#endif // LODEPNG_WRAPPER_HPP
