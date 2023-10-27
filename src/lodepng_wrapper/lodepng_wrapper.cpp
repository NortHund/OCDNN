// Wrapper created based on the example from the lodepng library
#include "lodepng_wrapper.h"

using namespace lodepng_wrapper;

LodepngWrapper::LodepngWrapper()
{
    ;
}

LodepngWrapper::~LodepngWrapper()
{
    free(_image);
}

unsigned LodepngWrapper::load_image(const char* filename)
{
    unsigned error = lodepng_decode32_file(&_image, &_width, &_height, filename);
    if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    _imageIsGrayscaled = false;
    return error;
}

unsigned LodepngWrapper::save_image(const char* filename)
{
    if(_imageIsGrayscaled) {
        return save_greyimage(filename);
    }
    else {
        return save_RGBAimage(filename);
    }

}
unsigned LodepngWrapper::save_RGBAimage(const char* filename)
{
    unsigned error = lodepng_encode_file(filename, _image, _width, _height, LCT_RGBA, 8);
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return error;
}

unsigned LodepngWrapper::save_greyimage(const char* filename)
{
    unsigned error = lodepng_encode_file(filename, _image, _width, _height, LCT_GREY, 8);
    if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    return error;
}

unsigned LodepngWrapper::transform_to_grayscale()
{
    // transform image buffer to greyscale with following formula: Y=0.2126R+0.7152G+0.0722B
    unsigned char *grey_image = (unsigned char*)malloc(_width * _height);

    for (unsigned i = 0; i < _width * _height * 4; i += 4)
    {
        unsigned char r = _image[i];
        unsigned char g = _image[i + 1];
        unsigned char b = _image[i + 2];
        unsigned char y = (unsigned char)(0.2126 * r + 0.7152 * g + 0.0722 * b);
        grey_image[i / 4] = y;
    }

    //reallocate image size and replace it with grey image
    _image = (unsigned char*)realloc(_image, _width * _height);
    memcpy(_image, grey_image, _width * _height);
    
    _imageIsGrayscaled = true;

    free(grey_image);

    return 0;
}

void LodepngWrapper::clone_image(unsigned char* dest)
{
    if (_imageIsGrayscaled) {
        memcpy(dest, _image, _width * _height);
    }
    else {
        memcpy(dest, _image, _width * _height * 4);
    }
}

void LodepngWrapper::set_image(unsigned char* src, unsigned width, unsigned height, int channels)
{
    this->_width = width;
    this->_height = height;
    if (_image != 0) {
        free(_image);
    }

    if (channels == GREY_CHANNELS) {
        _imageIsGrayscaled = true;
    }
    else {
        _imageIsGrayscaled = false;
    }
    
    _image = (unsigned char*)malloc(width * height * channels);
    memcpy(_image, src, width * height * channels);
}

unsigned LodepngWrapper::resize_image(unsigned scalingFactor)
{
    if (_imageIsGrayscaled) {
        unsigned int resized_width = _width / scalingFactor;
        unsigned int resized_height = _height / scalingFactor;
        
        // shrink the image
        unsigned char *resized_image = (unsigned char*)malloc(resized_width * resized_height);

        unsigned int ind = 0;
        unsigned int currow = 0;
        for (unsigned i = 0; i < (resized_width * resized_height); i++)
        {   
            resized_image[i] = _image[ind];
            ind = ind + scalingFactor;
            if (ind >= (_width + (currow * _width))) {
                currow = currow + scalingFactor;
                ind = (currow * _width);
            }
        }

        _image = (unsigned char*)realloc(_image, resized_width * resized_height);
        memcpy(_image, resized_image, resized_width * resized_height);

        _width = resized_width;
        _height = resized_height;

        free(resized_image);

        return 0;
    }

    return 1;
}

void LodepngWrapper::apply_filter(void (*filter)(unsigned char* image, unsigned width, unsigned height, unsigned windowSize), unsigned windowSize)
{
    filter(_image, _width, _height, windowSize);
}

void LodepngWrapper::occlusion_fill(void (*filter)(unsigned char *image, unsigned char *outImage, int width, int height))
{
    if (_imageIsGrayscaled) {
        unsigned char *occlusion_fill_image = (unsigned char*)malloc(_width * _height);
        filter(_image, occlusion_fill_image, _width, _height);
        memcpy(_image, occlusion_fill_image, _width * _height);
        free(occlusion_fill_image);
    }
}

unsigned LodepngWrapper::get_width()
{
    return _width;
}

unsigned LodepngWrapper::get_height()
{
    return _height;
}
