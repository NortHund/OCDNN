#include "image_convert.h"

int layer0w = 24;
int layer0h = 24;
int layer0d = 3;



double* matrixL0double;

int abft_err = 0;

int freememory() {
    free(matrixL0double);
}

static void createVectors()
{
    matrixL0double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));

    for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL0double[i] = 0;
    }

}

int load_image(const char* filename)
{
    std::vector<unsigned char> L0char;

    unsigned width;
    unsigned height;

    //unsigned output = lodepng_decode32_file(&L0char,&width, &height, filename);
    unsigned output = lodepng::decode(L0char, width, height, filename);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;


    /*for (int i = 0; i < 10; i++) {
        printf("%d ", L0char[i]);
    }
    printf("\n");*/

    for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL0double[i] = L0char[i];
    }

    return 1;
}

int main() {
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    int result = 0;

    createVectors();

    load_image("../../source-img/in0.png");

    for (int i=0; i <(layer0d * layer0h * layer0w) ; i++) {
        matrixL0double[i] = 1;
    }

    sw.saveEndPoint();
    //cleaning bufs and memory allocation
    freememory();

    //print opencl information
    printPlatformInfo(false);

    return 0;
}
