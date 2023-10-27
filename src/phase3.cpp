#include "phase3.h"

    lodepng_wrapper::LodepngWrapper img0;
    lodepng_wrapper::LodepngWrapper img1;
    lodepng_wrapper::LodepngWrapper combinedImage;

    int scaling_factor = 4;

    struct LoadImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.load_image("../../source-img/im0.png");
            error = img1.load_image("../../source-img/im1.png");
            return (int) error;
        }
    };

    struct ResizeImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.resize_image(scaling_factor);
            error = img1.resize_image(scaling_factor);
            return (int) error;
        }
    };
    
    struct TransformToGreyscale : public IProgram
    {
        int run() override
        {
            unsigned error = img0.transform_to_grayscale();
            error = img1.transform_to_grayscale();
            return (int) error;
        }
    };

    struct SaveGreyscaleImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.save_image("../../output-img/im0_grey.png");
            error = img1.save_image("../../output-img/im1_grey.png");
            return (int) error;
        }
    };

    struct SaveResizedImage : public IProgram
    {
        int run() override
        {
            unsigned error = img0.save_image("../../output-img/im0_grey_resized.png");
            error = img1.save_image("../../output-img/im1_grey_resized.png");
            return (int) error;
        }
    };

    struct ZNCCResizedImage : public IProgram
    {
        int run() override
        {
            unsigned char * t_img0 = (unsigned char*)malloc(img0.get_width() * img0.get_height());

            unsigned char * t_img1 = (unsigned char*)malloc(img1.get_width() *
                                                            img1.get_height());

            img0.clone_image(t_img0);
            img1.clone_image(t_img1);
            
            unsigned char *t_leftToRightImage = (unsigned char*)malloc(img0.get_width() *
                                                            img0.get_height());

            unsigned char *t_rightToLeftImage = (unsigned char*)malloc(img0.get_width() *
                                                            img0.get_height());

            // run the ZNCC
            OMP_ZNCCFilterOptimizedC(t_leftToRightImage, t_rightToLeftImage, t_img0, t_img1, img0.get_width(), img0.get_height(), 9);

            img0.set_image(t_leftToRightImage, img0.get_width(), img0.get_height(), GREY_CHANNELS);
            img1.set_image(t_rightToLeftImage, img1.get_width(), img1.get_height(), GREY_CHANNELS);
            
            img0.save_image("../../output-img/im0_grey_resized_zncc.png");
            img1.save_image("../../output-img/im1_grey_resized_zncc.png");
            
            free(t_img0);
            free(t_img1);
            free(t_leftToRightImage);
            free(t_rightToLeftImage);

            return 0;
        }
    };

    struct SaveZNCCImage : public IProgram
    {
        int run() override
        {
            img0.save_image("../../output-img/im0_grey_resized_zncc.png");
            img1.save_image("../../output-img/im1_grey_resized_zncc.png");
            return 0;
        }
    };

    struct CrosscheckImage : public IProgram
    {
        int run() override
        {
            unsigned char * t_img0 = (unsigned char*)malloc(img0.get_width() * img0.get_height());
                                                            
            unsigned char * t_img1 = (unsigned char*)malloc(img1.get_width() * img1.get_height());

            unsigned char * t_combinedImg = (unsigned char*)malloc(img1.get_width() * img1.get_height());

            img0.clone_image(t_img0);
            img1.clone_image(t_img1);

            crossCheckTwoImages(t_img0, t_img1, 50, t_combinedImg, img0.get_width() * img0.get_height());
            combinedImage.set_image(t_combinedImg, img0.get_width(), img0.get_height(), GREY_CHANNELS);

            free(t_img0);
            free(t_img1);
            free(t_combinedImg);

            return 0;
        }
    };

    struct SaveCrosscheckImage : public IProgram
    {
        int run() override
        {
            unsigned error = combinedImage.save_image("../../output-img/im_cc.png");
            return error;
        }
    };


    struct OcclusionFilterImage : public IProgram
    {
        int run() override
        {
            unsigned error = 0;

            combinedImage.occlusion_fill(fillZeroPixels);
            //img0.occlusion_fill(occFillOptimizedC);

            return (int) error;
        }
    };

    struct SaveOcclusionImage : public IProgram
    {
        int run() override
        {

            unsigned error = combinedImage.save_image("../../output-img/im_of.png");
            return (int) error;
        }
    };
    

int main()
{
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    // Step 3
    ProgramStopwatch Program_sw(clock);
    LoadImage loadImage;
    ResizeImage resizeImage;
    SaveResizedImage saveResizedImage;
    
    TransformToGreyscale transformToGreyscale;
    SaveGreyscaleImage saveGreyscaleImage;
    ZNCCResizedImage ZNCCResizedImage;
    CrosscheckImage crosscheckImage;
    OcclusionFilterImage occlusionFilterImage;
    
    SaveZNCCImage saveZNCCImage;
    SaveCrosscheckImage saveCrosscheckImage;
    SaveOcclusionImage saveOcclusionImage;

    int result = Program_sw.runProgram(loadImage);
    std::cout << "Load image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(transformToGreyscale);
    std::cout << "Transform to greyscale return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveGreyscaleImage);
    std::cout << "Save greyscale image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(resizeImage);
    std::cout << "Resize image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveResizedImage);
    std::cout << "Save resized image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(ZNCCResizedImage);
    std::cout << "ZNCC filter resized images return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveZNCCImage);
    std::cout << "Save ZNCC images return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(crosscheckImage);
    std::cout << "crosscheck image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveCrosscheckImage);
    std::cout << "Save crosschecked image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(occlusionFilterImage);
    std::cout << "Occlusion fill image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    result = Program_sw.runProgram(saveOcclusionImage);
    std::cout << "Save occlusion filled image return result: " << result << std::endl;
    std::cout << "Elapsed time: " << Program_sw.getElapsedTime() << " us" << std::endl;

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    printPlatformProfile(false);
    return 0;
}
