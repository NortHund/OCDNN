#include "phase6_main.h"
#include "phase6_model.h"
#include "phase6_ref.h"

int layer0w = 32;
int layer0h = 32;
int layer0d = 1;

int layer1w = 28;
int layer1h = 28;
int layer1d = 6;

int layer2w = LENGTH_FEATURE2;
int layer2h = LENGTH_FEATURE2;
int layer2d = LAYER2;

int layer3w = LENGTH_FEATURE3;
int layer3h = LENGTH_FEATURE3;
int layer3d = LAYER3;

int layer4w = LENGTH_FEATURE4;
int layer4h = LENGTH_FEATURE4;
int layer4d = LAYER4;

int layer5w = LENGTH_FEATURE5;
int layer5h = LENGTH_FEATURE5;
int layer5d = LAYER5;

int layer6w = OUTPUT;
int layer6h = 1;
int layer6d = 1;

int w01w = 5;
int w01h = 5;

int w12w = 5;
int w12h = 5;

int w23w = 5;
int w23h = 5;

int w34w = 5;
int w34h = 5;

int w45w = 5;
int w45h = 5;

int w56w = 5;
int w56h = 5;

double* matrixL0double;
double* matrixL1double;
double* matrixL2double;
double* matrixL3double;
double* matrixL4double;
double* matrixL5double;
double* matrixL6double;

double* matrixL1insum;
double* matrixL1outsum;

double* matrixL3insum;
double* matrixL3outsum;

double* matrixL5insum;
double* matrixL5outsum;

double* matrixL6insum;
double* matrixL6outsum;

double* matrixW01double;
double* matrixW12double;
double* matrixW23double;
double* matrixW34double;
double* matrixW45double;
double* matrixW56double;

double* matrixW01sum;
double* matrixW23sum;
double* matrixW45sum;
double* matrixW56sum;

double* matrixB01double;
double* matrixB12double;
double* matrixB23double;
double* matrixB34double;
double* matrixB45double;
double* matrixB56double;

double *ics;
double* ocs;
double* csc;

int abft_err = 0;
int abft_err_ind = 0;

static void forward_ocl(int abft)
{
    int abftflag = 0;
    int counter = 0;

    if (abft == 1) {
        ocl_phase2.setbufs_l01ics();
        ocl_phase2.convolution_nb(layer0w, layer0h, 1, w01w, w01h, layer1w, layer1h, 1, 0, 0);
        counter++;
        //printf("conv layer 1 ics convolutions: %d \n", counter);
    }

    //layer 1 convolution ocl
    ocl_phase2.setbufs_l01();
    //convolution
    counter = 0;
    for (int x = 0; x < (layer0d); ++x) {
        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.convolution_nb(layer0w, layer0h, layer0d, w01w, w01h, layer1w, layer1h, layer1d, x, y);
            counter++;
        }
    }
    //printf("conv layer 0-1 convolutions: %d \n", counter);

    if (abft == 1) {
        ocl_phase2.setbufs_l01ocs();
        ocl_phase2.output_sum_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, 1, 0, 0);

        //for (int i = 0; i < 1; i++) { csc[i] = 0; }
        ocl_phase2.setbufs_l01csc();
        ocl_phase2.cs_compare_nb(0, layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);

    }

    //ocs
    ocs[0] = 0;

    //layer1 relu
    ocl_phase2.setbuf_l1rb();

    for (int y = 0; y < layer1d; ++y) {
        ocl_phase2.relu_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, y);
    }

    if (abft == 1) {
        //layer 1 relu + bias doubling and compare
        ocl_phase2.setbuf_l1rbd();

        for (int y = 0; y < layer1d; ++y) {
            ocl_phase2.relu_nb(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, y);
        }

        ocl_phase2.setbufs_l1rbcsc();
        ocl_phase2.cs_compare_nb(1, layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0);
    }


    //layer 2 ocl max pooling
    ocl_phase2.setbuf_l12();

    for (int y = 0; y < layer2d; ++y) {
        ocl_phase2.maxpool_nb(layer1w, layer1h, layer1d, 2, 2, layer2w, layer2h, layer2d, 0, y);
    }

    if (abft == 1) {
        //layer 2 doubling and compare
        ocl_phase2.setbuf_l12d();

        for (int y = 0; y < layer2d; ++y) {
            ocl_phase2.maxpool_nb(layer1w, layer1h, layer1d, 2, 2, layer2w, layer2h, layer2d, 0, y);
        }

        ocl_phase2.setbufs_l2csc();
        ocl_phase2.cs_compare_nb(2, layer2w, layer2h, layer2d, w01w, w01h, layer2w, layer2h, layer2d, 0, 0);
    }

    if (abft == 1) {
        //layer 3 matrix ics:
        counter=0;
        ocl_phase2.setbufs_l23ics();
        for (int x = 0; x < (layer2d); ++x) {
            ocl_phase2.convolution_nb(layer2w, layer2h, 1, w01w, w01h, layer3w, layer3h, 1, x, 0);
            counter++;
        }
    }

    //layer3 convolution
    ocl_phase2.setbufs_l23();

    counter=0;
    for (int x = 0; x < (layer2d); ++x) {
        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.convolution_nb(layer2w, layer2h, layer2d, w23w, w23h, layer3w, layer3h, layer3d, x, y);
            counter++;
        }
    }
    //printf("conv layer 2-3 convolutions: %d \n",counter);

    if (abft == 1) {
        ocl_phase2.setbufs_l23ocs();
        ocl_phase2.output_sum_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, 1, 0, 0);

        ocl_phase2.setbufs_l23csc();
        ocl_phase2.cs_compare_nb(3, layer3w, layer3h, 1, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    }

    ocl_phase2.setbuf_l3rb();

    //layer3 relu
    for (int y = 0; y < layer3d; ++y) {
        ocl_phase2.relu_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, y);
    }

    if (abft == 1) {
        //layer 3 relu + bias doubling and compare
        ocl_phase2.setbuf_l3rbd();

        for (int y = 0; y < layer3d; ++y) {
            ocl_phase2.relu_nb(layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, y);
        }

        ocl_phase2.setbufs_l3rbcsc();
        ocl_phase2.cs_compare_nb(4, layer3w, layer3h, layer3d, w01w, w01h, layer3w, layer3h, layer3d, 0, 0);
    }

    //layer4 ocl max pooling
    ocl_phase2.setbuf_l34();
    for (int y = 0; y < layer4d; ++y) {
        ocl_phase2.maxpool_nb(layer3w, layer3h, layer3d, 2, 2, layer4w, layer4h, layer4d, 0, y);
    }

    if (abft == 1) {
        //layer 4 max pool doubling and compare
        ocl_phase2.setbuf_l4d();

        for (int y = 0; y < layer4d; ++y) {
            ocl_phase2.maxpool_nb(layer3w, layer3h, layer3d, 2, 2, layer4w, layer4h, layer4d, 0, y);
        }

        ocl_phase2.setbufs_l2csc();
        ocl_phase2.cs_compare_nb(5, layer4w, layer4h, layer4d, w01w, w01h, layer4w, layer4h, layer4d, 0, 0);
    }

    if (abft == 1) {
        //layer 5 matrix ics:
        ocl_phase2.setbufs_l45ics();
        counter =0;
        for (int x = 0; x < (layer4d); ++x) {
            ocl_phase2.convolution_nb(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, 1, x, 0);
            counter++;
        }
        //printf("conv layer 5 ics convolutions: %d \n",counter);
    }


    ocl_phase2.setbufs_l45();

    counter = 0;
    for (int x = 0; x < (layer4d); ++x) {
        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.convolution_nb(layer4w, layer4h, layer4d, w01w, w01h, layer5w, layer5h, layer5d, x, y);
            counter++;
        }
    }
    //printf("conv layer 4-5 convolutions: %d \n",counter);

    if (abft == 1) {
        //layer 5 output cs
        ocl_phase2.setbufs_l45ocs();
        ocl_phase2.output_sum_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, 1, 0, 0);

        ocl_phase2.setbufs_l45csc();
        ocl_phase2.cs_compare_nb(6, layer5w, layer5h, 1, w01w, w01h, layer5w, layer5h, 1, 0, 0);
        //there is a problem with reading and writing beyond index 4/5? to the csc or ocs csc buf
    }


    //layer5 relu
    ocl_phase2.setbuf_l5rb();
    for (int y = 0; y < layer5d; ++y) {
        ocl_phase2.relu_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, y);
    }

    if (abft == 1) {
        //layer 5 relu + bias doubling and compare
        ocl_phase2.setbuf_l5rbd();

        for (int y = 0; y < layer5d; ++y) {
            ocl_phase2.relu_nb(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, y);
        }

        ocl_phase2.setbufs_l5rbcsc();
        ocl_phase2.cs_compare_nb(7, layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0);
    }

    //ocl_phase2.last_read(layer5w, layer5h, layer5d, w01w, w01h, layer5w, layer5h, layer5d, 0, 0, matrixL5double);

    /*for (int i = 0; i < layer5d; ++i) {
        for (int j = 0; j < layer5h; ++j) {
            for (int k = 0; k < layer5w; ++k) {
                printf("%f ", matrixL5double[(i * layer5h * layer5w) + (j * layer5w) + k]);
            }
            printf("\n");
        }
    }*/

    //abft csc results:
    if (abft == 1) {
        ocl_phase2.cs_compare_read(layer1w, layer1h, layer1d, w01w, w01h, layer1w, layer1h, layer1d, 0, 0, csc);

        for (int i = 0; i < 10; i++) {
            if (csc[i] != 0) {
                abftflag = 1;
            }
        }

        if (abftflag == 1) {
            printf("csc: \n");
            for (int i = 0; i < 10; i++) {
                printf("%f ", csc[i]);
            }
            printf("\n");
        }
    }

    //output layer matrix multiplication in c++
    /*for (int y = 0; y < (OUTPUT); ++y) {
        matrixL6double[y] = 0;
    }

    //printf("L5 and W56: \n");

    for (int x = 0; x < (layer5d * layer5h * layer5w); ++x) {
        for (int y = 0; y < (OUTPUT); ++y) {
            matrixL6double[y] += matrixL5double[x] * matrixW56double[x * (OUTPUT) + y];
            //printf("W56: %f \n",matrixW56double[x * (OUTPUT) + y]);
        }
        //printf("L5: %f \n",matrixL5double[x]);
    }
    //printf("\n");*/

    if (abft == 1) {
        //flattened matrix ics
        ocl_phase2.setbuf_l56ics();
        ocl_phase2.flatmat_ics(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0);
        ocl_phase2.flatmat_ics_read(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0,
                                matrixL6insum);
    }
    //printf("l6insum: \n");
    //printf(" %f", matrixL6insum[0]);
    //printf("\n");

    //flattened matrix multiplication
    ocl_phase2.setbuf_l56();
    ocl_phase2.flatmat_nb(layer5w, layer5h, layer5d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0);

    ocl_phase2.last_read(layer6w, layer6h, layer6d, w01w, w01h, layer6w, layer6h, layer6d, 0, 0, matrixL6double);

    ocs[0] = 0;

    if (abft == 1) {

        //printf("l6 ics: %f\n", ics[0]);

        //printf("L6: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ",matrixL6double[i]);
            ocs[0] += matrixL6double[i];
        }
        //printf("\n");

        //printf("l6 ics with bias: %f\n", ics[0]);
        //printf("l6 ocs: %f\n", ocs[0]);

        if (abs(matrixL6insum[0] - ocs[0]) > 0.00001) {
            printf("l6 ics: %f\n", matrixL6insum[0]);
            printf("l6 ocs: %f\n", ocs[0]);
            abftflag = 1;
        }
    }

    for (int j = 0; j < (OUTPUT); ++j) {
        if (matrixL6double[j] < 0) {
            matrixL6double[j] = 0;
        }
    }

    /*for (int j = 0; j < (OUTPUT); ++j) {
        if (matrixL6double[j] + matrixB56double[j] > 0) {
            matrixL6double[j] += matrixB56double[j];
        } else {
            matrixL6double[j] = 0;
        }
    }*/

    /*printf("L6: ");
    for (uint8 i = 1; i < OUTPUT; ++i) {
        printf("%f ",matrixL6double[i]);
    }
    printf("\n");*/
    //printf("abft flag: %d\n", abftflag);
    abft_err = abftflag;

}



static inline void load_input_ocl(image input)
{
    const long sz = sizeof(image) / sizeof(**input);
    double mean = 0, std = 0;
    for (int i = 0; i < (sizeof(image) / sizeof(*input)); i++) {
        for (int j = 0; j < (sizeof(*input) / sizeof(**input)); j++) {
            mean += input[i][j];
            std += input[i][j] * input[i][j];
        }
    }

    mean /= sz;
    std = sqrt(std / sz - mean * mean);
    double testIn = 0;
    for (int i = 0; i < layer1h; i++) {
        for (int j = 0; j < layer1w; j++) {
            testIn = (input[i][j] - mean) / std;
            matrixL0double[(i + PADDING) * layer0w + (j + PADDING)] = testIn;
        }
    }
}





uint8 Predict_ocl(image input, int abft, uint8 count)
{

    load_input_ocl(input);
    ocl_phase2.write_image(matrixL0double);

    forward_ocl(abft);

    //print L6 floats here
    //printf("layer6 Ocl: ");
    for (uint8 i = 0; i < OUTPUT; ++i) {
        //printf("%f ", matrixL6double[i]);
    }
    //printf("\n");

    //getting result from the output matrix/vector
    const int outlen = OUTPUT;
    uint8 result = 0;
    double maxvalue = 0;
    for (uint8 i = 0; i < OUTPUT; ++i) {
        if (matrixL6double[i] > maxvalue) {
            maxvalue = matrixL6double[i];
            result = i;
        }
    }

    if (abft_err != 0) {
        printf("abft flag raised\n");
        abft_err = 0;
    }

    //printf("ocl prediction: %d\n",result);
    return result;
}



int init_mem(int abft) {
    zero_vectors();
    ocl_phase2.write_layer(matrixL1double, matrixL2double, matrixL3double, matrixL4double, matrixL5double, matrixL6double);

    if (abft == 1) {
        ocl_phase2.write_layersums(matrixL1insum, matrixL1outsum, matrixL3insum, matrixL3outsum, matrixL5insum, matrixL5outsum, matrixL6insum, csc);
        ocl_phase2.write_layerdoubles(matrixL1double, matrixL2double, matrixL3double, matrixL4double, matrixL5double);
    }

}




int testing_ocl(image *test_data, uint8 *test_label, int abft, int total_size)
{
    int right = 0, percent = 0;
    int acctemp = 0;
    for (int i = 0; i < total_size; ++i)
    {
        init_mem(abft);
        uint8 l = test_label[i];
        int p = Predict_ocl(test_data[i], 1, 10);
        //printf("prediction ocl: %d \n", p);
        right += l == p;
        acctemp += l == p;
        if (i % 100 == 0 && i != 0) {
            printf("ocl accuracy: %d / 100\n", acctemp);
            printf("ocl total accuracy: %d / %d \n", right, i);
            acctemp = 0;
        }
        /*if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);*/
    }
    return right;
}

int testing_comb(LeNet5 *lenet, image *test_data, uint8 *test_label, int abft, int total_size)
{
    int right = 0, percent = 0;
    int acctemp = 0;
    for (int i = 0; i < total_size; ++i)
    {
        init_mem(abft);
        uint8 l = test_label[i];
        int p = Predict_ocl(test_data[i], abft, 10);
        int pp = Predict(lenet, test_data[i], 10);
        //printf("prediction ocl: %d \n", p);

        if (p != pp) {
            printf("Prediction mismatch! ocl:%d, c++:%d \n", p, pp);
        }

        right += l == p;
        acctemp += l == p;
        if (i % 100 == 0 && i != 0) {
            printf("ocl accuracy: %d / 100\n", acctemp);
            printf("ocl total accuracy: %d / %d \n", right, i);
            acctemp = 0;
        }
        /*if (i * 100 / total_size > percent)
            printf("test:%2d%%\n", percent = i * 100 / total_size);*/
    }
    return right;
}

struct Predict_noabft : public IProgram {
    int run() override {

        forward_ocl(0);

        //print L6 floats here
        //printf("layer6 Ocl: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ", matrixL6double[i]);
        }
        //printf("\n");

        //getting result from the output matrix/vector
        const int outlen = OUTPUT;
        uint8 result = 0;
        double maxvalue = 0;
        for (uint8 i = 0; i < OUTPUT; ++i) {
            if (matrixL6double[i] > maxvalue) {
                maxvalue = matrixL6double[i];
                result = i;
            }
        }

        //printf("ocl prediction: %d\n",result);
        return result;
    }
};

struct Predict_abft : public IProgram {
    int run() override {

        forward_ocl(1);

        //print L6 floats here
        //printf("layer6 Ocl: ");
        for (uint8 i = 0; i < OUTPUT; ++i) {
            //printf("%f ", matrixL6double[i]);
        }
        //printf("\n");

        //getting result from the output matrix/vector
        const int outlen = OUTPUT;
        uint8 result = 0;
        double maxvalue = 0;
        for (uint8 i = 0; i < OUTPUT; ++i) {
            if (matrixL6double[i] > maxvalue) {
                maxvalue = matrixL6double[i];
                result = i;
            }
        }

        if (abft_err != 0) {
            printf("abft flag raised\n");
            abft_err = 0;
        }

        //printf("ocl prediction: %d\n",result);
        return result;
    }
};

int main() {
    // Measure total time
    ChronoClock clock;
    Stopwatch sw(clock);

    sw.saveStartPoint();

    //Start clock
    ProgramStopwatch Program_sw(clock);

    //Program
    Predict_noabft predictNoabft;
    Predict_abft predictAbft;

    int result = 0;

    image *test_data = (image *) calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *) calloc(COUNT_TEST, sizeof(uint8));
    if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
        printf("ERROR!\nDataset File Not Found! Please Copy Dataset to the Folder Including the exe\n");
        free(test_data);
        free(test_label);
        system("pause");
    }


    LeNet5 *lenet = (LeNet5 *) malloc(sizeof(LeNet5));
    if (load(lenet, LENET_FILE))
        Initial(lenet);

    createVectors();
    copyModel(lenet);

    int abft_enable = 1;

    ocl_phase2.write_weights(matrixW01double, matrixW23double, matrixW45double, matrixW56double);
    ocl_phase2.write_bias(matrixB01double, matrixB23double, matrixB45double, matrixB56double);

    if (abft_enable == 1) {
        ocl_phase2.write_weightsums(matrixW01sum, matrixW23sum, matrixW45sum, matrixW56sum);
    }
    double time1 = 0;
    double time2 = 0;

    load_input_ocl(test_data[1]);
    ocl_phase2.write_image(matrixL0double);

    for (int i= 0; i < 1; i++) {
        init_mem(0);
        result = Program_sw.runProgram(predictNoabft);
        time1 += Program_sw.getElapsedTime();
    }
    std::cout << "single prediction without abft: " << result << std::endl;
    std::cout << "Elapsed time: " << time1 << " us" << std::endl;
    load_input_ocl(test_data[1]);
    ocl_phase2.write_image(matrixL0double);

    for (int i= 0; i < 1; i++) {
        init_mem(1);
        result = Program_sw.runProgram(predictAbft);
        time2 += Program_sw.getElapsedTime();
    }
    std::cout << "single prediction with abft: " << result << std::endl;
    std::cout << "Elapsed time: " << time2 << " us" << std::endl;
    std::cout << "Time difference: " << time2 - time1 << " us" << std::endl;
    std::cout << "abft overhead: " << ((time2 - time1) / time1) << " " << std::endl;
    

    //int right = testing(lenet, test_data, test_label, COUNT_TEST);
    //int right = testing(lenet, test_data, test_label, 100);
    //printf("c++ right: %d / 100 \n", right);

    //int right_ocl = testing_ocl(test_data, test_label, 100);
    //printf("ocl accuracy: %d / %d \n", right_ocl, 100);

    for (int i = 0; i < COUNT_TEST; i++) {
        int right_comb = testing_comb(lenet, test_data, test_label, abft_enable,  COUNT_TEST);
        printf("accuracy: %d / %d \n", right_comb, COUNT_TEST);
    }
    // p = Predict(lenet, test_data[120], 10);
    //int oclp = Predict_ocl(test_data[120], 10);
    //printf("c: %d, ocl: %d \n",p, oclp);

    //printf("%d/%d\n", right, COUNT_TEST);
    free(lenet);
    free(test_data);
    free(test_label);

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    ocl_phase2.print_kernel_execution_times();

    ocl_phase2.free_bufs();

    free(matrixL0double);
    free(matrixL1double);
    free(matrixL2double);
    free(matrixL3double);
    free(matrixL4double);
    free(matrixL5double);
    free(matrixL6double);

    free(matrixL1insum);
    free(matrixL1outsum);

    free(matrixL3insum);
    free(matrixL3outsum);

    free(matrixL5insum);
    free(matrixL5outsum);

    free(matrixL6insum);
    free(matrixL6outsum);

    free(matrixW01double);
    free(matrixW12double);
    free(matrixW23double);
    free(matrixW34double);
    free(matrixW45double);
    free(matrixW56double);

    free(matrixW01sum);
    free(matrixW23sum);
    free(matrixW45sum);
    free(matrixW56sum);

    free(matrixB01double);
    free(matrixB12double);
    free(matrixB23double);
    free(matrixB34double);
    free(matrixB45double);
    free(matrixB56double);

    free(ics);
    free(ocs);
    free(csc);

    printPlatformInfo(false);
    return 0;
}
