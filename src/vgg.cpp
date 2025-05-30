#include "vgg.h"
#include <string>

int layer0w = 224;
int layer0h = 224;
int layer0d = 3;

int c1w = 224;
int c1h = 224;
int c1d = 64;
int c10d = 3;

int c2w = 112;
int c2h = 112;
int c2d = 128;
int c20d = 64;

int c3w = 56;
int c3h = 56;
int c3d = 256;
int c30d = 128;

int c4w = 28;
int c4h = 28;
int c4d = 512;
int c40d = 256;

int c5w = 14;
int c5h = 14;
int c5d = 512;
int c50d = 512;

int c6w = 7;
int c6h = 7;
int c6d = 512;

int k1 = 3;
int k2 = 3;
int k3 = 3;
int k4 = 3;
int k5 = 3;


int c1pad = 1;
int c2pad = 1;
int c3pad = 1;
int c4pad = 1;
int c5pad = 1;

double* matrixL00double;
double* matrixL01double;
double* matrixL02double;
double* matrixL03double;
double* matrixL04double;
double* matrixL05double;
double* matrixL06double;
double* matrixL07double;
double* matrixL08double;
double* matrixL09double;

double* matrixW11double;
double* matrixW12double;
double* matrixW21double;
double* matrixW22double;
double* matrixW31double;
double* matrixW32double;
double* matrixW33double;
double* matrixW41double;
double* matrixW42double;
double* matrixW43double;
double* matrixW51double;
double* matrixW52double;
double* matrixW53double;
double* matrixW61double;
double* matrixW62double;
double* matrixW63double;

double* matrixB11double;
double* matrixB12double;
double* matrixB21double;
double* matrixB22double;
double* matrixB31double;
double* matrixB32double;
double* matrixB33double;
double* matrixB41double;
double* matrixB42double;
double* matrixB43double;
double* matrixB51double;
double* matrixB52double;
double* matrixB53double;
double* matrixB61double;
double* matrixB62double;
double* matrixB63double;

double* matrixW11sum;
double* matrixW12sum;
double* matrixW21sum;
double* matrixW22sum;
double* matrixW31sum;
double* matrixW32sum;
double* matrixW33sum;
double* matrixW41sum;
double* matrixW42sum;
double* matrixW43sum;
double* matrixW51sum;
double* matrixW52sum;
double* matrixW53sum;
double* matrixW61sum;
double* matrixW62sum;
double* matrixW63sum;

double* matrixB11sum;
double* matrixB12sum;
double* matrixB21sum;
double* matrixB22sum;
double* matrixB31sum;
double* matrixB32sum;
double* matrixB33sum;
double* matrixB41sum;
double* matrixB42sum;
double* matrixB43sum;
double* matrixB51sum;
double* matrixB52sum;
double* matrixB53sum;
double* matrixB61sum;
double* matrixB62sum;
double* matrixB63sum;

double* ics;
double* ocs;
double* csc;

double* matrixR;
double* matrixR2;
double* matrixR3;
double* matrixR4;
double* matrixR5;
double* matrixR6;
double* matrixResult0;
double* matrixResult1;
double* matrixResult2;
double* matrixResult3;
double* matrixResult4;
double* matrixResult5;
double* matrixResult6;
double* matrixResult7;
double* matrixResult8;
double* matrixResult9;

int* ref_prediction;

int abft_err = 0;

int abft_error_flag = 0;

int total_output_errors = 0;
int total_sig_output_errors = 0;
int total_layer_output_errors = 0;
int total_sig_layer_output_errors = 0;
int total_prediction_error = 0;
int total_abft_errors = 0;
int total_abft_inference_errors = 0;
int total_false_negatives = 0;
int total_sig_false_negatives = 0;

int freememory() {
    free(matrixL00double);
    free(matrixL01double);
    free(matrixL02double);
    free(matrixL03double);
    free(matrixL04double);
    free(matrixL05double);
    free(matrixL06double);
    free(matrixL07double);
    free(matrixL08double);
    free(matrixL09double);

    free(matrixW11double);
    free(matrixW12double);
    free(matrixW21double);
    free(matrixW22double);
    free(matrixW31double);
    free(matrixW32double);
    free(matrixW33double);
    free(matrixW41double);
    free(matrixW42double);
    free(matrixW43double);
    free(matrixW51double);
    free(matrixW52double);
    free(matrixW53double);
    free(matrixW61double);
    free(matrixW62double);
    free(matrixW63double);

    free(matrixB11double);
    free(matrixB12double);
    free(matrixB21double);
    free(matrixB22double);
    free(matrixB31double);
    free(matrixB32double);
    free(matrixB33double);
    free(matrixB41double);
    free(matrixB42double);
    free(matrixB43double);
    free(matrixB51double);
    free(matrixB52double);
    free(matrixB53double);
    free(matrixB61double);
    free(matrixB62double);
    free(matrixB63double);

    free(matrixR);
    free(matrixR2);
    free(matrixR3);
    free(matrixR4);
    free(matrixR5);
    free(matrixR6);
    free(matrixResult0);
    free(matrixResult1);
    free(matrixResult2);
    free(matrixResult3);
    free(matrixResult4);
    free(matrixResult5);
    free(matrixResult6);
    free(matrixResult7);
    free(matrixResult8);
    free(matrixResult9);

    free(ref_prediction);

    free(matrixW11sum);
    free(matrixW12sum);
    free(matrixW21sum);
    free(matrixW22sum);
    free(matrixW31sum);
    free(matrixW32sum);
    free(matrixW33sum);
    free(matrixW41sum);
    free(matrixW42sum);
    free(matrixW43sum);
    free(matrixW51sum);
    free(matrixW52sum);
    free(matrixW53sum);
    free(matrixW61sum);
    free(matrixW62sum);
    free(matrixW63sum);

    free(matrixB11sum);
    free(matrixB12sum);
    free(matrixB21sum);
    free(matrixB22sum);
    free(matrixB31sum);
    free(matrixB32sum);
    free(matrixB33sum);
    free(matrixB41sum);
    free(matrixB42sum);
    free(matrixB43sum);
    free(matrixB51sum);
    free(matrixB52sum);
    free(matrixB53sum);
    free(matrixB61sum);
    free(matrixB62sum);
    free(matrixB63sum);

    free(ics);
    free(ocs);
    free(csc);
}

static void createVectors()
{
    matrixL00double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL01double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL02double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL03double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL04double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL05double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL06double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL07double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL08double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));
    matrixL09double = (double*)malloc((layer0d) * (layer0w * layer0h) * sizeof(double));

    matrixW11double = (double*)malloc((layer0d) * (c1d) * (k1 * k1) * sizeof(double));
    matrixW12double = (double*)malloc((c1d) * (c1d) * (k1 * k1) * sizeof(double));
    matrixW21double = (double*)malloc((c20d) * (c2d) * (k2 * k2) * sizeof(double));
    matrixW22double = (double*)malloc((c2d) * (c2d) * (k2 * k2) * sizeof(double));
    matrixW31double = (double*)malloc((c30d) * (c3d) * (k3 * k3) * sizeof(double));
    matrixW32double = (double*)malloc((c3d) * (c3d) * (k3 * k3) * sizeof(double));
    matrixW33double = (double*)malloc((c3d) * (c3d) * (k3 * k3) * sizeof(double));
    matrixW41double = (double*)malloc((c40d) * (c4d) * (k4 * k4) * sizeof(double));
    matrixW42double = (double*)malloc((c4d) * (c4d) * (k4 * k4) * sizeof(double));
    matrixW43double = (double*)malloc((c4d) * (c4d) * (k4 * k4) * sizeof(double));
    matrixW51double = (double*)malloc((c50d) * (c5d) * (k5 * k5) * sizeof(double));
    matrixW52double = (double*)malloc((c5d) * (c5d) * (k5 * k5) * sizeof(double));
    matrixW53double = (double*)malloc((c5d) * (c5d) * (k5 * k5) * sizeof(double));
    matrixW61double = (double*)malloc((4096*25088) * sizeof(double));
    matrixW62double = (double*)malloc((4096*4096) * sizeof(double));
    matrixW63double = (double*)malloc((1000*4096) * sizeof(double));

    matrixB11double = (double*)malloc((c1d) * sizeof(double));
    matrixB12double = (double*)malloc((c1d) * sizeof(double));
    matrixB21double = (double*)malloc((c2d) * sizeof(double));
    matrixB22double = (double*)malloc((c2d) * sizeof(double));
    matrixB31double = (double*)malloc((c3d) * sizeof(double));
    matrixB32double = (double*)malloc((c3d) * sizeof(double));
    matrixB33double = (double*)malloc((c3d) * sizeof(double));
    matrixB41double = (double*)malloc((c4d) * sizeof(double));
    matrixB42double = (double*)malloc((c4d) * sizeof(double));
    matrixB43double = (double*)malloc((c4d) * sizeof(double));
    matrixB51double = (double*)malloc((c5d) * sizeof(double));
    matrixB52double = (double*)malloc((c5d) * sizeof(double));
    matrixB53double = (double*)malloc((c5d) * sizeof(double));
    matrixB61double = (double*)malloc((4096) * sizeof(double));
    matrixB62double = (double*)malloc((4096) * sizeof(double));
    matrixB63double = (double*)malloc((1000) * sizeof(double));

    matrixW11sum = (double*)malloc((layer0d) * (k1 * k1) * sizeof(double));
    matrixW12sum = (double*)malloc((c1d) * (k1 * k1) * sizeof(double));
    matrixW21sum = (double*)malloc((c1d) * (k2 * k2) * sizeof(double));
    matrixW22sum = (double*)malloc((c2d) * (k2 * k2) * sizeof(double));
    matrixW31sum = (double*)malloc((c2d) * (k3 * k3) * sizeof(double));
    matrixW32sum = (double*)malloc((c3d) * (k3 * k3) * sizeof(double));
    matrixW33sum = (double*)malloc((c3d) * (k3 * k3) * sizeof(double));
    matrixW41sum = (double*)malloc((c3d) * (k3 * k3) * sizeof(double));
    matrixW42sum = (double*)malloc((c4d) * (k3 * k3) * sizeof(double));
    matrixW43sum = (double*)malloc((c4d) * (k3 * k3) * sizeof(double));
    matrixW51sum = (double*)malloc((c5d) * (k3 * k3) * sizeof(double));
    matrixW52sum = (double*)malloc((c5d) * (k3 * k3) * sizeof(double));
    matrixW53sum = (double*)malloc((c5d) * (k3 * k3) * sizeof(double));
    matrixW61sum = (double*)malloc((25088) * sizeof(double));
    matrixW62sum = (double*)malloc((4096) * sizeof(double));
    matrixW63sum = (double*)malloc((4096) * sizeof(double));

    matrixB11sum = (double*)malloc(sizeof(double));
    matrixB12sum = (double*)malloc(sizeof(double));
    matrixB21sum = (double*)malloc(sizeof(double));
    matrixB22sum = (double*)malloc(sizeof(double));
    matrixB31sum = (double*)malloc(sizeof(double));
    matrixB32sum = (double*)malloc(sizeof(double));
    matrixB33sum = (double*)malloc(sizeof(double));
    matrixB41sum = (double*)malloc(sizeof(double));
    matrixB42sum = (double*)malloc(sizeof(double));
    matrixB43sum = (double*)malloc(sizeof(double));
    matrixB51sum = (double*)malloc(sizeof(double));
    matrixB52sum = (double*)malloc(sizeof(double));
    matrixB53sum = (double*)malloc(sizeof(double));
    matrixB61sum = (double*)malloc(sizeof(double));
    matrixB62sum = (double*)malloc(sizeof(double));
    matrixB63sum = (double*)malloc(sizeof(double));

    ics = (double*)malloc((c1w * c1h) * sizeof(double));
    ocs = (double*)malloc((c1w * c1h) * sizeof(double));
    csc = (double*)malloc((37) * sizeof(double));
    for (int i = 0; i < (37); i++) {
        csc[i] = 0;
    }

    matrixR = (double*)malloc((c1d) * (c1w * c1h) * sizeof(double));
    matrixR2 = (double*)malloc((c2d) * (c2w * c2h) * sizeof(double));
    matrixR3 = (double*)malloc((c3d) * (c3w * c3h) * sizeof(double));
    matrixR4 = (double*)malloc((c4d) * (c4w * c4h) * sizeof(double));
    matrixR5 = (double*)malloc((c5d) * (c5w * c5h) * sizeof(double));
    matrixR6 = (double*)malloc(1000 * sizeof(double));
    matrixResult0 = (double*)malloc(1000 * sizeof(double));
    matrixResult1 = (double*)malloc(1000 * sizeof(double));
    matrixResult2 = (double*)malloc(1000 * sizeof(double));
    matrixResult3 = (double*)malloc(1000 * sizeof(double));
    matrixResult4 = (double*)malloc(1000 * sizeof(double));
    matrixResult5 = (double*)malloc(1000 * sizeof(double));
    matrixResult6 = (double*)malloc(1000 * sizeof(double));
    matrixResult7 = (double*)malloc(1000 * sizeof(double));
    matrixResult8 = (double*)malloc(1000 * sizeof(double));
    matrixResult9 = (double*)malloc(1000 * sizeof(double));

    ref_prediction = (int*)malloc(10 * sizeof(int));

    for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL00double[i] = 0;
        matrixL01double[i] = 0;
        matrixL02double[i] = 0;
        matrixL03double[i] = 0;
        matrixL04double[i] = 0;
        matrixL05double[i] = 0;
        matrixL06double[i] = 0;
        matrixL07double[i] = 0;
        matrixL08double[i] = 0;
        matrixL09double[i] = 0;
    }

}

static void save_result() {
    //in0
    FILE *fp = fopen(("../../source-data/result0.bin"), "wb");
    fwrite(matrixResult0, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in1
    fp = fopen(("../../source-data/result1.bin"), "wb");
    fwrite(matrixResult1, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in2
    fp = fopen(("../../source-data/result2.bin"), "wb");
    fwrite(matrixResult2, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in3
    fp = fopen(("../../source-data/result3.bin"), "wb");
    fwrite(matrixResult3, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in4
    fp = fopen(("../../source-data/result4.bin"), "wb");
    fwrite(matrixResult4, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in5
    fp = fopen(("../../source-data/result5.bin"), "wb");
    fwrite(matrixResult5, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in6
    fp = fopen(("../../source-data/result6.bin"), "wb");
    fwrite(matrixResult6, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in7
    fp = fopen(("../../source-data/result7.bin"), "wb");
    fwrite(matrixResult7, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in8
    fp = fopen(("../../source-data/result8.bin"), "wb");
    fwrite(matrixResult8, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in9
    fp = fopen(("../../source-data/result9.bin"), "wb");
    fwrite(matrixResult9, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //reference predictions
    fp = fopen(("../../source-data/predictions.bin"), "wb");
    fwrite(ref_prediction, (10) * sizeof(int), 1, fp);
    fclose(fp);

}

static void load_result() {
    //in0
    FILE *fp = fopen("../../source-data/result0.bin", "rb");
    fread(matrixResult0, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in1
    fp = fopen("../../source-data/result1.bin", "rb");
    fread(matrixResult1, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in2
    fp = fopen("../../source-data/result2.bin", "rb");
    fread(matrixResult2, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in3
    fp = fopen("../../source-data/result3.bin", "rb");
    fread(matrixResult3, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in4
    fp = fopen("../../source-data/result4.bin", "rb");
    fread(matrixResult4, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in5
    fp = fopen("../../source-data/result5.bin", "rb");
    fread(matrixResult5, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in6
    fp = fopen("../../source-data/result6.bin", "rb");
    fread(matrixResult6, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in7
    fp = fopen("../../source-data/result7.bin", "rb");
    fread(matrixResult7, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in8
    fp = fopen("../../source-data/result8.bin", "rb");
    fread(matrixResult8, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //in9
    fp = fopen("../../source-data/result9.bin", "rb");
    fread(matrixResult9, (1000) * sizeof(double), 1, fp);
    fclose(fp);

    //reference predictions
    fp = fopen(("../../source-data/predictions.bin"), "rb");
    fread(ref_prediction, (10) * sizeof(int), 1, fp);
    fclose(fp);
}

static void copyModel() {
    //1-1
    FILE *fp = fopen("../../source-data/vgg_weights/0.bin", "rb");
    fread(matrixW11double, 1728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/1.bin", "rb");
    fread(matrixB11double, 64 * sizeof(double), 1, fp);
    fclose(fp);

    //1-2
    fp = fopen("../../source-data/vgg_weights/2.bin", "rb");
    fread(matrixW12double, 36864 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/3.bin", "rb");
    fread(matrixB12double, 64 * sizeof(double), 1, fp);
    fclose(fp);

    //2-1
    fp = fopen("../../source-data/vgg_weights/4.bin", "rb");
    fread(matrixW21double, 73728 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/5.bin", "rb");
    fread(matrixB21double, 128 * sizeof(double), 1, fp);
    fclose(fp);

    //2-2
    fp = fopen("../../source-data/vgg_weights/6.bin", "rb");
    fread(matrixW22double, 147456 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/7.bin", "rb");
    fread(matrixB22double, 128 * sizeof(double), 1, fp);
    fclose(fp);

    //3-1
    fp = fopen("../../source-data/vgg_weights/8.bin", "rb");
    fread(matrixW31double, 294912 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/9.bin", "rb");
    fread(matrixB31double, 256 * sizeof(double), 1, fp);
    fclose(fp);

    //3-2
    fp = fopen("../../source-data/vgg_weights/10.bin", "rb");
    fread(matrixW32double, 589824 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/11.bin", "rb");
    fread(matrixB32double, 256 * sizeof(double), 1, fp);
    fclose(fp);

    //3-3
    fp = fopen("../../source-data/vgg_weights/12.bin", "rb");
    fread(matrixW33double, 589824 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/13.bin", "rb");
    fread(matrixB33double, 256 * sizeof(double), 1, fp);
    fclose(fp);

    //4-1
    fp = fopen("../../source-data/vgg_weights/14.bin", "rb");
    fread(matrixW41double, 1179648 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/15.bin", "rb");
    fread(matrixB41double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //4-2
    fp = fopen("../../source-data/vgg_weights/16.bin", "rb");
    fread(matrixW42double, 2359296 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/17.bin", "rb");
    fread(matrixB42double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //4-3
    fp = fopen("../../source-data/vgg_weights/18.bin", "rb");
    fread(matrixW43double, 2359296 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/19.bin", "rb");
    fread(matrixB43double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //5-1
    fp = fopen("../../source-data/vgg_weights/20.bin", "rb");
    fread(matrixW51double, 2359296 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/21.bin", "rb");
    fread(matrixB51double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //5-2
    fp = fopen("../../source-data/vgg_weights/22.bin", "rb");
    fread(matrixW52double, 2359296 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/23.bin", "rb");
    fread(matrixB52double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //5-3
    fp = fopen("../../source-data/vgg_weights/24.bin", "rb");
    fread(matrixW53double, 2359296 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/25.bin", "rb");
    fread(matrixB53double, 512 * sizeof(double), 1, fp);
    fclose(fp);

    //6-1
    fp = fopen("../../source-data/vgg_weights/26.bin", "rb");
    fread(matrixW61double, 102760448 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/27.bin", "rb");
    fread(matrixB61double, 4096 * sizeof(double), 1, fp);
    fclose(fp);

    //6-2
    fp = fopen("../../source-data/vgg_weights/28.bin", "rb");
    fread(matrixW62double, 16777216 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/29.bin", "rb");
    fread(matrixB62double, 4096 * sizeof(double), 1, fp);
    fclose(fp);

    //6-3
    fp = fopen("../../source-data/vgg_weights/30.bin", "rb");
    fread(matrixW63double, 4096000 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_weights/31.bin", "rb");
    fread(matrixB63double, 1000 * sizeof(double), 1, fp);
    fclose(fp);

}

static void copyWeightSums() {
    //1-1
    FILE *fp = fopen("../../source-data/vgg_sums/0s.bin", "rb");
    fread(matrixW11sum, (3*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/1s.bin", "rb");
    fread(matrixB11sum, sizeof(double), 1, fp);
    fclose(fp);

    //1-2
    fp = fopen("../../source-data/vgg_sums/2s.bin", "rb");
    fread(matrixW12sum, (64*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/3s.bin", "rb");
    fread(matrixB12sum, sizeof(double), 1, fp);
    fclose(fp);

    //2-1
    fp = fopen("../../source-data/vgg_sums/4s.bin", "rb");
    fread(matrixW21sum, (64*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/5s.bin", "rb");
    fread(matrixB21sum, sizeof(double), 1, fp);
    fclose(fp);

    //2-2
    fp = fopen("../../source-data/vgg_sums/6s.bin", "rb");
    fread(matrixW22sum, (128*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/7s.bin", "rb");
    fread(matrixB22sum, sizeof(double), 1, fp);
    fclose(fp);

    //3-1
    fp = fopen("../../source-data/vgg_sums/8s.bin", "rb");
    fread(matrixW31sum, (128*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/9s.bin", "rb");
    fread(matrixB31sum, sizeof(double), 1, fp);
    fclose(fp);

    //3-2
    fp = fopen("../../source-data/vgg_sums/10s.bin", "rb");
    fread(matrixW32sum, (256*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/11s.bin", "rb");
    fread(matrixB32sum, sizeof(double), 1, fp);
    fclose(fp);

    //3-3
    fp = fopen("../../source-data/vgg_sums/12s.bin", "rb");
    fread(matrixW33sum, (256*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/13s.bin", "rb");
    fread(matrixB33sum, sizeof(double), 1, fp);
    fclose(fp);

    //4-1
    fp = fopen("../../source-data/vgg_sums/14s.bin", "rb");
    fread(matrixW41sum, (256*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/15s.bin", "rb");
    fread(matrixB41sum, sizeof(double), 1, fp);
    fclose(fp);

    //4-2
    fp = fopen("../../source-data/vgg_sums/16s.bin", "rb");
    fread(matrixW42sum, (512*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/17s.bin", "rb");
    fread(matrixB42sum, sizeof(double), 1, fp);
    fclose(fp);

    //4-3
    fp = fopen("../../source-data/vgg_sums/18s.bin", "rb");
    fread(matrixW43sum, (512*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/19s.bin", "rb");
    fread(matrixB43sum, sizeof(double), 1, fp);
    fclose(fp);

    //5-1
    fp = fopen("../../source-data/vgg_sums/20s.bin", "rb");
    fread(matrixW51sum, (512*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/21s.bin", "rb");
    fread(matrixB51sum, sizeof(double), 1, fp);
    fclose(fp);

    //5-2
    fp = fopen("../../source-data/vgg_sums/22s.bin", "rb");
    fread(matrixW52sum, (512*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/23s.bin", "rb");
    fread(matrixB52sum, sizeof(double), 1, fp);
    fclose(fp);

    //5-3
    fp = fopen("../../source-data/vgg_sums/24s.bin", "rb");
    fread(matrixW53sum, (512*3*3) * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/25s.bin", "rb");
    fread(matrixB53sum, sizeof(double), 1, fp);
    fclose(fp);

    //6-1
    fp = fopen("../../source-data/vgg_sums/26s.bin", "rb");
    fread(matrixW61sum, 25088 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/27s.bin", "rb");
    fread(matrixB61sum, sizeof(double), 1, fp);
    fclose(fp);

    //6-2
    fp = fopen("../../source-data/vgg_sums/28s.bin", "rb");
    fread(matrixW62sum, 4096 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/29s.bin", "rb");
    fread(matrixB62sum, sizeof(double), 1, fp);
    fclose(fp);

    //6-3
    fp = fopen("../../source-data/vgg_sums/30s.bin", "rb");
    fread(matrixW63sum, 4096 * sizeof(double), 1, fp);
    fclose(fp);

    fp = fopen("../../source-data/vgg_sums/31s.bin", "rb");
    fread(matrixB63sum, sizeof(double), 1, fp);
    fclose(fp);
}

static void create_weight_sums() {

    for (int i = 0; i < (64 * k1 * k1); i++) {
        matrixW21sum[i] = 0;
    }


    matrixB21sum[0] = 0;


    for (int h = 0; h < 64; h++) {
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < k1; j++) {
                for (int k = 0; k < k1; k++) {
                    matrixW21sum[(h * k1 * k1) + (j * k1) + k] += matrixW21double[(h * c1d * k1 * k1) + (i * k1 * k1) + (j * k1) + k];
                }
            }
        }
    }

    for (int h = 0; h < c2d; h++) {
        matrixB21sum[0] += matrixB21double[h];
    }


}



// Get kernel execution time in microseconds
unsigned long get_kernel_execution_time(cl_event &event, cl_command_queue &command_queue)
{
    clFinish(command_queue);

    cl_ulong time_start;
    cl_ulong time_end;

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    return (time_end - time_start) / 1000;
}

class OCL
{
public:
    OCL()
    {
        _ocl_base.reset(new OCL_Base());

        init_programs();
        init_kernels();
        create_layers();
        create_bufs_abft(csc);
    }

    ~OCL()
    {
        free_bufs();
    }

    void init_programs()
    {
        prog_cv = _ocl_base->CreateProgramFromFile("kernels/vgg-conv.cl");
        prog_util = _ocl_base->CreateProgramFromFile("kernels/vgg-util.cl");
    }

    void init_kernels()
    {
        _ocl_base->CreateKernelFromProgram(prog_cv, "convolution_double"); //0
        _ocl_base->CreateKernelFromProgram(prog_cv, "input_sum"); //1
        _ocl_base->CreateKernelFromProgram(prog_cv, "output_sum"); //2
        _ocl_base->CreateKernelFromProgram(prog_cv, "cs_compare"); //3
        _ocl_base->CreateKernelFromProgram(prog_util, "relu"); //4
        _ocl_base->CreateKernelFromProgram(prog_util, "maxpool"); //5
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat"); //6
        _ocl_base->CreateKernelFromProgram(prog_util, "relu_d"); //7
        _ocl_base->CreateKernelFromProgram(prog_util, "maxpool_d"); //8
        _ocl_base->CreateKernelFromProgram(prog_util, "flatmat_ics"); //9
        _ocl_base->CreateKernelFromProgram(prog_cv, "convolution_ic"); //10
        _ocl_base->CreateKernelFromProgram(prog_cv, "output_r"); //11
    }

    cl_mem l0Buffer = nullptr;

    cl_mem c11Buf = nullptr;
    cl_mem c12Buf = nullptr;
    cl_mem c21Buf = nullptr;
    cl_mem c22Buf = nullptr;
    cl_mem c31Buf = nullptr;
    cl_mem c32Buf = nullptr;
    cl_mem c41Buf = nullptr;
    cl_mem c42Buf = nullptr;
    cl_mem c51Buf = nullptr;
    cl_mem c52Buf = nullptr;
    cl_mem c61Buf = nullptr;
    cl_mem c62Buf = nullptr;
    cl_mem c63Buf = nullptr;
    cl_mem c6rBuf = nullptr;

    cl_mem w11Buffer = nullptr;
    cl_mem w12Buffer = nullptr;
    cl_mem w21Buffer = nullptr;
    cl_mem w22Buffer = nullptr;
    cl_mem w31Buffer = nullptr;
    cl_mem w32Buffer = nullptr;
    cl_mem w33Buffer = nullptr;
    cl_mem w41Buffer = nullptr;
    cl_mem w42Buffer = nullptr;
    cl_mem w43Buffer = nullptr;
    cl_mem w51Buffer = nullptr;
    cl_mem w52Buffer = nullptr;
    cl_mem w53Buffer = nullptr;
    cl_mem w61Buffer = nullptr;
    cl_mem w62Buffer = nullptr;
    cl_mem w63Buffer = nullptr;

    cl_mem b11Buffer = nullptr;
    cl_mem b12Buffer = nullptr;
    cl_mem b21Buffer = nullptr;
    cl_mem b22Buffer = nullptr;
    cl_mem b31Buffer = nullptr;
    cl_mem b32Buffer = nullptr;
    cl_mem b33Buffer = nullptr;
    cl_mem b41Buffer = nullptr;
    cl_mem b42Buffer = nullptr;
    cl_mem b43Buffer = nullptr;
    cl_mem b51Buffer = nullptr;
    cl_mem b52Buffer = nullptr;
    cl_mem b53Buffer = nullptr;
    cl_mem b61Buffer = nullptr;
    cl_mem b62Buffer = nullptr;
    cl_mem b63Buffer = nullptr;

    cl_mem w11sBuffer = nullptr;
    cl_mem w12sBuffer = nullptr;
    cl_mem w21sBuffer = nullptr;
    cl_mem w22sBuffer = nullptr;
    cl_mem w31sBuffer = nullptr;
    cl_mem w32sBuffer = nullptr;
    cl_mem w33sBuffer = nullptr;
    cl_mem w41sBuffer = nullptr;
    cl_mem w42sBuffer = nullptr;
    cl_mem w43sBuffer = nullptr;
    cl_mem w51sBuffer = nullptr;
    cl_mem w52sBuffer = nullptr;
    cl_mem w53sBuffer = nullptr;
    cl_mem w61sBuffer = nullptr;
    cl_mem w62sBuffer = nullptr;
    cl_mem w63sBuffer = nullptr;

    cl_mem b11sBuffer = nullptr;
    cl_mem b12sBuffer = nullptr;
    cl_mem b21sBuffer = nullptr;
    cl_mem b22sBuffer = nullptr;
    cl_mem b31sBuffer = nullptr;
    cl_mem b32sBuffer = nullptr;
    cl_mem b33sBuffer = nullptr;
    cl_mem b41sBuffer = nullptr;
    cl_mem b42sBuffer = nullptr;
    cl_mem b43sBuffer = nullptr;
    cl_mem b51sBuffer = nullptr;
    cl_mem b52sBuffer = nullptr;
    cl_mem b53sBuffer = nullptr;
    cl_mem b61sBuffer = nullptr;
    cl_mem b62sBuffer = nullptr;
    cl_mem b63sBuffer = nullptr;

    cl_mem c1dBuf = nullptr;
    cl_mem c2dBuf = nullptr;
    cl_mem c3dBuf = nullptr;
    cl_mem c4dBuf = nullptr;
    cl_mem c5dBuf = nullptr;
    cl_mem c6dBuf = nullptr;

    cl_mem icsBuf = nullptr;
    cl_mem ficBuf = nullptr;
    cl_mem fisBuf = nullptr;
    cl_mem cicBuf = nullptr;
    cl_mem cisrBuf = nullptr;
    cl_mem cisBuf = nullptr;
    cl_mem ocsrBuf = nullptr;
    cl_mem ocsBuf = nullptr;
    cl_mem cscBuf = nullptr;
    cl_mem csczBuf = nullptr;

    unsigned create_layers()
    {
        c11Buf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  c1d * c1h * c1w * sizeof(double),
                                  nullptr,
                                  NULL);

        c12Buf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE,
                                  c1d * c1h * c1w * sizeof(double),
                                  nullptr,
                                  NULL);

        c21Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2d * c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        c22Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2d * c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        c31Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c3d * c3h * c3w * sizeof(double),
                                nullptr,
                                NULL);

        c32Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c3d * c3h * c3w * sizeof(double),
                                nullptr,
                                NULL);

        c41Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c4d * c4h * c4w * sizeof(double),
                                nullptr,
                                NULL);

        c42Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c4d * c4h * c4w * sizeof(double),
                                nullptr,
                                NULL);

        c51Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c5d * c5h * c5w * sizeof(double),
                                nullptr,
                                NULL);

        c52Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c5d * c5h * c5w * sizeof(double),
                                nullptr,
                                NULL);

        c61Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                25088 * sizeof(double),
                                nullptr,
                                NULL);

        c62Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                4096 * sizeof(double),
                                nullptr,
                                NULL);

        c63Buf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                4096 * sizeof(double),
                                nullptr,
                                NULL);

        c6rBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                1000 * sizeof(double),
                                nullptr,
                                NULL);

    }

    unsigned create_bufs_abft(double* cscptr)
    {
        c1dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c1d * c1h * c1w * sizeof(double),
                                nullptr,
                                NULL);

        c2dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c2d * c2h * c2w * sizeof(double),
                                nullptr,
                                NULL);

        c3dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c3d * c3h * c3w * sizeof(double),
                                nullptr,
                                NULL);

        c4dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c4d * c4h * c4w * sizeof(double),
                                nullptr,
                                NULL);

        c5dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c5d * c5h * c5w * sizeof(double),
                                nullptr,
                                NULL);

        c6dBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                4096 * sizeof(double),
                                nullptr,
                                NULL);

        icsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c1h * c1w * sizeof(double),
                                nullptr,
                                NULL);

        ficBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                4096 * sizeof(double),
                                nullptr,
                                NULL);

        fisBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                10 * sizeof(double),
                                nullptr,
                                NULL);

        cicBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28 * 512) * sizeof(double),
                                nullptr,
                                NULL);

        cisrBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28 * 64) * sizeof(double), // todo check dimensions
                                nullptr,
                                NULL);

        cisBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28) * sizeof(double),
                                nullptr,
                                NULL);


        ocsBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                c1h * c1w * sizeof(double),
                                nullptr,
                                NULL);

        ocsrBuf = clCreateBuffer(_ocl_base->context,
                                CL_MEM_READ_WRITE,
                                (28 * 28 * 64) * sizeof(double),
                                nullptr,
                                NULL);

        /*cscBuf = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  37 * sizeof(double),
                                  csc,
                                  NULL);*/

        cscBuf = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   37 * sizeof(double),
                                   cscptr,
                                   NULL);
    }

    unsigned zero_CSC()
    {
        cscBuf = clCreateBuffer(_ocl_base->context,
                                        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        37 * sizeof(double),
                                        csc,
                                        NULL);
    }

    unsigned free_bufs()
    {
        clReleaseMemObject(l0Buffer);

        clReleaseMemObject(c11Buf);
        clReleaseMemObject(c12Buf);
        clReleaseMemObject(c21Buf);
        clReleaseMemObject(c22Buf);
        clReleaseMemObject(c31Buf);
        clReleaseMemObject(c32Buf);
        clReleaseMemObject(c41Buf);
        clReleaseMemObject(c42Buf);
        clReleaseMemObject(c51Buf);
        clReleaseMemObject(c52Buf);
        clReleaseMemObject(c61Buf);
        clReleaseMemObject(c62Buf);
        clReleaseMemObject(c63Buf);
        clReleaseMemObject(c6rBuf);

        clReleaseMemObject(w11Buffer);
        clReleaseMemObject(w12Buffer);
        clReleaseMemObject(w21Buffer);
        clReleaseMemObject(w22Buffer);
        clReleaseMemObject(w31Buffer);
        clReleaseMemObject(w32Buffer);
        clReleaseMemObject(w33Buffer);
        clReleaseMemObject(w41Buffer);
        clReleaseMemObject(w42Buffer);
        clReleaseMemObject(w43Buffer);
        clReleaseMemObject(w51Buffer);
        clReleaseMemObject(w52Buffer);
        clReleaseMemObject(w53Buffer);
        clReleaseMemObject(w61Buffer);
        clReleaseMemObject(w62Buffer);
        clReleaseMemObject(w63Buffer);

        clReleaseMemObject(b11Buffer);
        clReleaseMemObject(b12Buffer);
        clReleaseMemObject(b21Buffer);
        clReleaseMemObject(b22Buffer);
        clReleaseMemObject(b31Buffer);
        clReleaseMemObject(b32Buffer);
        clReleaseMemObject(b33Buffer);
        clReleaseMemObject(b41Buffer);
        clReleaseMemObject(b42Buffer);
        clReleaseMemObject(b43Buffer);
        clReleaseMemObject(b51Buffer);
        clReleaseMemObject(b52Buffer);
        clReleaseMemObject(b53Buffer);
        clReleaseMemObject(b61Buffer);
        clReleaseMemObject(b62Buffer);
        clReleaseMemObject(b63Buffer);

        clReleaseMemObject(w11sBuffer);
        clReleaseMemObject(w12sBuffer);
        clReleaseMemObject(w21sBuffer);
        clReleaseMemObject(w22sBuffer);
        clReleaseMemObject(w31sBuffer);
        clReleaseMemObject(w32sBuffer);
        clReleaseMemObject(w33sBuffer);
        clReleaseMemObject(w41sBuffer);
        clReleaseMemObject(w42sBuffer);
        clReleaseMemObject(w43sBuffer);
        clReleaseMemObject(w51sBuffer);
        clReleaseMemObject(w52sBuffer);
        clReleaseMemObject(w53sBuffer);
        clReleaseMemObject(w61sBuffer);
        clReleaseMemObject(w62sBuffer);
        clReleaseMemObject(w63sBuffer);

        clReleaseMemObject(b11sBuffer);
        clReleaseMemObject(b12sBuffer);
        clReleaseMemObject(b21sBuffer);
        clReleaseMemObject(b22sBuffer);
        clReleaseMemObject(b31sBuffer);
        clReleaseMemObject(b32sBuffer);
        clReleaseMemObject(b33sBuffer);
        clReleaseMemObject(b41sBuffer);
        clReleaseMemObject(b42sBuffer);
        clReleaseMemObject(b43sBuffer);
        clReleaseMemObject(b51sBuffer);
        clReleaseMemObject(b52sBuffer);
        clReleaseMemObject(b53sBuffer);
        clReleaseMemObject(b61sBuffer);
        clReleaseMemObject(b62sBuffer);
        clReleaseMemObject(b63sBuffer);

        clReleaseMemObject(c1dBuf);
        clReleaseMemObject(c2dBuf);
        clReleaseMemObject(c3dBuf);
        clReleaseMemObject(c4dBuf);
        clReleaseMemObject(c5dBuf);
        clReleaseMemObject(c6dBuf);

        clReleaseMemObject(icsBuf);
        clReleaseMemObject(ficBuf);
        clReleaseMemObject(fisBuf);
        clReleaseMemObject(cicBuf);
        clReleaseMemObject(cisrBuf);
        clReleaseMemObject(cisBuf);
        clReleaseMemObject(ocsBuf);
        clReleaseMemObject(cscBuf);
    }

    unsigned write_image(double* l0ptr) {
        l0Buffer = clCreateBuffer(_ocl_base->context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  layer0d * layer0h * layer0w * sizeof(double),
                                  l0ptr,
                                  NULL);
    }

    unsigned write_weights(double* w11ptr, double* w12ptr,
                           double* w21ptr, double* w22ptr,
                           double* w31ptr, double* w32ptr, double* w33ptr,
                           double* w41ptr, double* w42ptr, double* w43ptr,
                           double* w51ptr, double* w52ptr, double* w53ptr,
                           double* w61ptr, double* w62ptr, double* w63ptr)
    {
        w11Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   layer0d * c1d * k1 * k1 * sizeof(double),
                                   w11ptr,
                                   NULL);

        w12Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c1d * c1d * k1 * k1 * sizeof(double),
                                   w12ptr,
                                   NULL);

        w21Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * c1d * k1 * k1 * sizeof(double),
                                   w21ptr,
                                   NULL);

        w22Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * c2d * k1 * k1 * sizeof(double),
                                   w22ptr,
                                   NULL);

        w31Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * c2d * k3 * k3 * sizeof(double),
                                   w31ptr,
                                   NULL);

        w32Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * c3d * k3 * k3 * sizeof(double),
                                   w32ptr,
                                   NULL);

        w33Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * c3d * k3 * k3 * sizeof(double),
                                   w33ptr,
                                   NULL);

        w41Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * c3d * k4 * k4 * sizeof(double),
                                   w41ptr,
                                   NULL);

        w42Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * c4d * k4 * k4 * sizeof(double),
                                   w42ptr,
                                   NULL);

        w43Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * c4d * k4 * k4 * sizeof(double),
                                   w43ptr,
                                   NULL);

        w51Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * c4d * k5 * k5 * sizeof(double),
                                   w51ptr,
                                   NULL);

        w52Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * c5d * k5 * k5 * sizeof(double),
                                   w52ptr,
                                   NULL);

        w53Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * c5d * k5 * k5 * sizeof(double),
                                   w53ptr,
                                   NULL);

        w61Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   4096 * 25088 * sizeof(double),
                                   w61ptr,
                                   NULL);

        w62Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   4096 * 4096 * sizeof(double),
                                   w62ptr,
                                   NULL);

        w63Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   1000 * 4096 * sizeof(double),
                                   w63ptr,
                                   NULL);

    }

    unsigned write_bias(double* b11ptr, double* b12ptr,
                        double* b21ptr, double* b22ptr,
                        double* b31ptr, double* b32ptr, double* b33ptr,
                        double* b41ptr, double* b42ptr, double* b43ptr,
                        double* b51ptr, double* b52ptr, double* b53ptr,
                        double* b61ptr, double* b62ptr, double* b63ptr)
    {
        b11Buffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c1d * sizeof(double),
                                    b11ptr,
                                    NULL);

        b12Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c1d * sizeof(double),
                                   b12ptr,
                                   NULL);

        b21Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * sizeof(double),
                                   b21ptr,
                                   NULL);

        b22Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * sizeof(double),
                                   b22ptr,
                                   NULL);


        b31Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * sizeof(double),
                                   b31ptr,
                                   NULL);

        b32Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * sizeof(double),
                                   b32ptr,
                                   NULL);

        b33Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c3d * sizeof(double),
                                   b33ptr,
                                   NULL);

        b41Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * sizeof(double),
                                   b41ptr,
                                   NULL);

        b42Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * sizeof(double),
                                   b42ptr,
                                   NULL);

        b43Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c4d * sizeof(double),
                                   b43ptr,
                                   NULL);

        b51Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * sizeof(double),
                                   b51ptr,
                                   NULL);

        b52Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * sizeof(double),
                                   b52ptr,
                                   NULL);

        b53Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c5d * sizeof(double),
                                   b53ptr,
                                   NULL);

        b61Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   4096 * sizeof(double),
                                   b61ptr,
                                   NULL);

        b62Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   4096 * sizeof(double),
                                   b62ptr,
                                   NULL);

        b63Buffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   1000 * sizeof(double),
                                   b63ptr,
                                   NULL);
    }

    unsigned write_weight_sums(double* w11sptr, double* w12sptr,
                               double* w21sptr, double* w22sptr,
                               double* w31sptr, double* w32sptr, double* w33sptr,
                               double* w41sptr, double* w42sptr, double* w43sptr,
                               double* w51sptr, double* w52sptr, double* w53sptr,
                               double* w61sptr, double* w62sptr, double* w63sptr,
                               double* b11sptr, double* b12sptr,
                               double* b21sptr, double* b22sptr,
                               double* b31sptr, double* b32sptr, double* b33sptr,
                               double* b41sptr, double* b42sptr, double* b43sptr,
                               double* b51sptr, double* b52sptr, double* b53sptr,
                               double* b61sptr, double* b62sptr, double* b63sptr)
    {
        w11sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c10d * k1 * k1 * sizeof(double),
                                    w11sptr,
                                    NULL);

        w12sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c1d * k1 * k1 * sizeof(double),
                                    w12sptr,
                                    NULL);

        w21sBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c1d * k1 * k1 * sizeof(double),
                                   w21sptr,
                                   NULL);

        w22sBuffer = clCreateBuffer(_ocl_base->context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   c2d * k1 * k1 * sizeof(double),
                                   w22sptr,
                                   NULL);

        w31sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c2d * k3 * k3 * sizeof(double),
                                    w31sptr,
                                    NULL);

        w32sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c3d * k3 * k3 * sizeof(double),
                                    w32sptr,
                                    NULL);

        w33sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c3d * k3 * k3 * sizeof(double),
                                    w33sptr,
                                    NULL);

        w41sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c3d * k3 * k3 * sizeof(double),
                                    w41sptr,
                                    NULL);

        w42sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c4d * k3 * k3 * sizeof(double),
                                    w42sptr,
                                    NULL);

        w43sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c4d * k3 * k3 * sizeof(double),
                                    w43sptr,
                                    NULL);

        w51sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c5d * k3 * k3 * sizeof(double),
                                    w51sptr,
                                    NULL);

        w52sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c5d * k3 * k3 * sizeof(double),
                                    w52sptr,
                                    NULL);

        w53sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    c5d * k3 * k3 * sizeof(double),
                                    w53sptr,
                                    NULL);

        w61sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    25088 * sizeof(double),
                                    w61sptr,
                                    NULL);

        w62sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    4096 * sizeof(double),
                                    w62sptr,
                                    NULL);

        w63sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    4096 * sizeof(double),
                                    w63sptr,
                                    NULL);


        //biases
        b11sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b11sptr,
                                    NULL);

        b12sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b12sptr,
                                    NULL);

        b21sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b21sptr,
                                    NULL);

        b22sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b22sptr,
                                    NULL);

        b31sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b31sptr,
                                    NULL);

        b32sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b32sptr,
                                    NULL);

        b33sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b33sptr,
                                    NULL);


        b41sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b41sptr,
                                    NULL);

        b42sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b42sptr,
                                    NULL);

        b43sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b43sptr,
                                    NULL);


        b51sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b51sptr,
                                    NULL);

        b52sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b52sptr,
                                    NULL);

        b53sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b53sptr,
                                    NULL);


        b61sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b61sptr,
                                    NULL);

        b62sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b62sptr,
                                    NULL);

        b63sBuffer = clCreateBuffer(_ocl_base->context,
                                    CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(double),
                                    b63sptr,
                                    NULL);
    }

    double buf_read(int ow, int oh, int od, double* optr, cl_mem bptr)
    {
        //Reading result from GPU memory to main memory
        cl_int status = clEnqueueReadBuffer(_ocl_base->commandQueue,
                                            bptr,
                                            0,
                                            0,
                                            od * ow * oh * sizeof(double),
                                            optr,
                                            0,
                                            nullptr,
                                            &_event);

        kernel_execution_times[1] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

    }

    unsigned convolution3(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, int iw, int ih, int id, int k, int pad, int ow, int oh, int od)
    {
        cl_int status;
        //Setting kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(0), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 1, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 2, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 3, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 6, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 7, sizeof(int), &k);
        status = clSetKernelArg(_ocl_base->GetKernel(0), 8, sizeof(int), &pad);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(0),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution3_ics(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, int iw, int ih, int id, int k, int pad, int ow, int oh, int od)
    {
        cl_int status;
        //Setting kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(10), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 1, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 2, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 3, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 4, sizeof(int), &id);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 5, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 6, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 7, sizeof(int), &k);
        status = clSetKernelArg(_ocl_base->GetKernel(10), 8, sizeof(int), &pad);

        size_t global_work_size[3];
        global_work_size[0] = iw;
        global_work_size[1] = ih;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(10),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[0] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned convolution3_abft(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icsbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                               int iw, int ih, int id, int k, int pad, int ow, int oh, int od, int cscInd)
    {
        //ics
        convolution3(ibuf, wsbuffer, bsbuf, icsbuffer, iw, ih, id, k, pad, ow, oh, 1);

        //convolution layer
        convolution3(ibuf, wbuf, bbuf, obuf, iw, ih, id, k, pad, ow, oh, od);

        //ocs
        output_sum(obuf, ocsbuffer, od, ow, oh);

        //csc
        cs_compare(icsbuffer, ocsbuffer, cscBuf, ow, oh, 1, cscInd);

        return 1;
    }

    unsigned convolution3_abft_icr(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icbuffer, cl_mem isrbuffer, cl_mem isbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                               int iw, int ih, int id, int k, int pad, int ow, int oh, int od, int cscInd)
    {
        //ic
        convolution3_ics(ibuf, wsbuffer, bsbuf, cicBuf, iw, ih, id, k, pad, ow, oh, od);

        //isr
        output_reduce(cicBuf, cisrBuf, 16, ow, oh, (id / 16));

        //is
        output_sum(cisrBuf, cisBuf, (id / 16), ow, oh);

        //output_sum(cicBuf, cisBuf, id, iw, ih);

        //convolution layer
        convolution3(ibuf, wbuf, bbuf, obuf, iw, ih, id, k, pad, ow, oh, od);

        //ocsr
        output_reduce(obuf, ocsrBuf, 16, ow, oh, (od / 16));

        //ocs
        output_sum(ocsrBuf, ocsbuffer, (od / 16), ow, oh);
        //output_sum(obuf, ocsbuffer, od, ow, oh);

        //csc
        cs_compare(cisBuf, ocsbuffer, cscBuf, ow, oh, 1, cscInd);

        return 1;
    }

    unsigned relu(cl_mem ibuf, cl_mem obuf, int ow, int oh, int od)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(4), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(4), 1, sizeof(cl_mem), (void *) &obuf);

        size_t global_work_size[3]; //and here
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(4),
                                        3, //was this the issue? ugh
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned relu_d(cl_mem ibuf, cl_mem obuf, int ow, int oh, int od)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(7), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(7), 1, sizeof(cl_mem), (void *) &obuf);

        size_t global_work_size[3]; //and here
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(7),
                                        3, //was this the issue? ugh
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned relu_dmr(cl_mem ibuf, cl_mem dbuf, cl_mem obuf, int ow, int oh, int od, int cscInd)
    {
        relu(ibuf, obuf, ow, oh, od);
        relu_d(ibuf, dbuf, ow, oh, od);

        cs_compare(obuf, dbuf, cscBuf, ow, oh, od, cscInd);

        return 1;
    }

    unsigned maxpool(cl_mem ibuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(5), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 2, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 3, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 4, sizeof(int), &kernel_size);
        status = clSetKernelArg(_ocl_base->GetKernel(5), 5, sizeof(int), &stride);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(5),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned maxpool_d(cl_mem ibuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh)
    {
        cl_int status;

        //Setting buffers to kernel arguments
        status = clSetKernelArg(_ocl_base->GetKernel(8), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 2, sizeof(int), &ih);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 3, sizeof(int), &iw);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 4, sizeof(int), &kernel_size);
        status = clSetKernelArg(_ocl_base->GetKernel(8), 5, sizeof(int), &stride);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = id;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(8),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned maxpool_dmr(cl_mem ibuf, cl_mem dbuf, cl_mem obuf, int iw, int ih, int id, int stride, int kernel_size, int ow, int oh, int cscInd)
    {
        maxpool(ibuf, obuf, iw, ih, id, stride, kernel_size, ow, oh);
        maxpool_d(ibuf, dbuf, iw, ih, id, stride, kernel_size, ow, oh);

        cs_compare(obuf, dbuf, cscBuf, ow, oh, id, cscInd);

        return 1;
    }

    unsigned output_sum(cl_mem ibuf, cl_mem obuf, int id, int ow, int oh)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(2), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(2), 2, sizeof(int), &id);

        size_t global_work_size[2];
        global_work_size[0] = ow;
        global_work_size[1] = oh;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(2),
                                        2,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned output_reduce(cl_mem ibuf, cl_mem obuf, int id, int ow, int oh, int od)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(11), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(11), 2, sizeof(int), &id);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(11),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned cs_compare(cl_mem ibuf, cl_mem obuf, cl_mem csbuf, int ow, int oh, int od, int csInd)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(3), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 2, sizeof(cl_mem), (void *) &csbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(3), 3, sizeof(int), &csInd);

        size_t global_work_size[3];
        global_work_size[0] = ow;
        global_work_size[1] = oh;
        global_work_size[2] = od;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(3),
                                        3,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[6] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf,
                     int iw, int ow)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(6), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 2, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 3, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(6), 4, sizeof(int), &iw);

        size_t global_work_size[1];
        global_work_size[0] = ow;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(6),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat_ics(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf,
                     int iw, int ow)
    {
        cl_int status;
        status = clSetKernelArg(_ocl_base->GetKernel(9), 0, sizeof(cl_mem), (void *) &ibuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 1, sizeof(cl_mem), (void *) &obuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 2, sizeof(cl_mem), (void *) &wbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 3, sizeof(cl_mem), (void *) &bbuf);
        status = clSetKernelArg(_ocl_base->GetKernel(9), 4, sizeof(int), &iw);

        size_t global_work_size[1];
        global_work_size[0] = ow;

        //Enqueueing kernel
        status = clEnqueueNDRangeKernel(_ocl_base->commandQueue,
                                        _ocl_base->GetKernel(9),
                                        1,
                                        NULL,
                                        global_work_size,
                                        NULL,
                                        0,
                                        NULL,
                                        &_event);
        if (status != CL_SUCCESS) {
            std::cerr << "ERROR: " <<  getErrorString(status)  << std::endl;
            exit(EXIT_FAILURE);
        }

        //kernel_execution_times[4] = get_kernel_execution_time(_event, _ocl_base->commandQueue);

        return (unsigned)status;
    }

    unsigned flatmat_abft(cl_mem ibuf, cl_mem wbuf, cl_mem bbuf, cl_mem obuf, cl_mem icbuffer, cl_mem isbuffer, cl_mem wsbuffer, cl_mem bsbuf, cl_mem ocsbuffer,
                          int iw, int ow, int cscInd)
    {
        //ic
        flatmat_ics(ibuf, wsbuffer, bsbuf, icbuffer, 32, (iw / 32));

        //is
        output_sum(icbuffer, isbuffer, (iw / 32), 1, 1);

        //matmul
        flatmat(ibuf, wbuf, bbuf, obuf, iw, ow);

        //ocs
        output_sum(obuf, ocsbuffer, ow, 1, 1);

        //csc
        cs_compare(isbuffer, ocsbuffer, cscBuf, 1, 1, 1, cscInd);

        return 1;
    }

    void print_kernel_execution_times()
    {
        std::cout << "OpenCL kernel execution times:\n";
        std::cout << "  Convolution: " << kernel_execution_times[0] << " us\n";
        std::cout << "  Convolution read: " << kernel_execution_times[1] << " us\n";
        std::cout << "  Output sum: " << kernel_execution_times[4] << " us\n";
        std::cout << "  Output sum read: " << kernel_execution_times[5] << " us\n";
        std::cout << "  Checksum compare: " << kernel_execution_times[6] << " us\n";
        std::cout << "  Checksum compare read: " << kernel_execution_times[7] << " us\n";
        std::cout << "\n\n";
    }

    std::unique_ptr<OCL_Base> _ocl_base;

private:
    cl_program prog_cv;
    cl_program prog_util;

    cl_event _event;

    // 0 - convolution
    // 1 - convolution read
    // 2 - input_sum
    // 3 - input_sum_read
    // 4 - ocs
    // 5 - ocs read
    // 6 - csc
    // 7 - csc
    // 8 - relu
    // 9 - relu read
    unsigned long kernel_execution_times[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
};

OCL ocl;

int load_images(const char* filename0, const char* filename1, const char* filename2,
                const char* filename3, const char* filename4, const char* filename5,
                const char* filename6, const char* filename7, const char* filename8,
                const char* filename9)
{
    std::vector<unsigned char> L00char;
    std::vector<unsigned char> L01char;
    std::vector<unsigned char> L02char;
    std::vector<unsigned char> L03char;
    std::vector<unsigned char> L04char;
    std::vector<unsigned char> L05char;
    std::vector<unsigned char> L06char;
    std::vector<unsigned char> L07char;
    std::vector<unsigned char> L08char;
    std::vector<unsigned char> L09char;

    unsigned width;
    unsigned height;

    //unsigned output = lodepng_decode32_file(&L0char,&width, &height, filename);
    unsigned output = lodepng::decode(L00char, width, height, filename0);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L01char, width, height, filename1);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L02char, width, height, filename2);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L03char, width, height, filename3);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L04char, width, height, filename4);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L05char, width, height, filename5);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L06char, width, height, filename6);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L07char, width, height, filename7);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L08char, width, height, filename8);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    output = lodepng::decode(L09char, width, height, filename9);
    if (output) std::cout << "decoder error " << output << ": " << lodepng_error_text(output) << std::endl;

    /*for (int i = 0; i < (layer0d * layer0h * layer0w); i++) {
        matrixL00double[i] = L00char[i];
    }*/

    //converting image to format used by the model
    for (int i = 0; i < (layer0d); i++) {
        for (int j = 0; j < (layer0h); j++) {
            for (int k = 0; k < (layer0w); k++) {
                    matrixL00double[(i * c1h * c1w) + (j * c1w) + k] = L00char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL01double[(i * c1h * c1w) + (j * c1w) + k] = L01char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL02double[(i * c1h * c1w) + (j * c1w) + k] = L02char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL03double[(i * c1h * c1w) + (j * c1w) + k] = L03char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL04double[(i * c1h * c1w) + (j * c1w) + k] = L04char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL05double[(i * c1h * c1w) + (j * c1w) + k] = L05char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL06double[(i * c1h * c1w) + (j * c1w) + k] = L06char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL07double[(i * c1h * c1w) + (j * c1w) + k] = L07char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL08double[(i * c1h * c1w) + (j * c1w) + k] = L08char[(j * 4 * c1w) + (k * 4) + i];
                    matrixL09double[(i * c1h * c1w) + (j * c1w) + k] = L09char[(j * 4 * c1w) + (k * 4) + i];

            }
        }
    }

    return 1;
}



int normalize_images(double* matrix)
{
    const long sz = 3*224*224;
    double mean = 0, std = 0;
    double val = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 224; j++) {
            for (int k = 0; k < 224; k++) {
                val = matrix[(i * c1h * c1w) + (j * c1w) + k];
                mean += val;
                std += val * val;
            }
        }
    }

    mean /= sz;
    std = sqrt(std / sz - mean * mean);
    double testIn = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 224; j++) {
            for (int k = 0; k < 224; k++) {
                testIn = (matrix[(i * c1h * c1w) + (j * c1w) + k] - mean) / std;
                matrix[(i * c1h * c1w) + (j * c1w) + k] = testIn;
            }
        }
    }

}

static void load_inputs() {
    load_images("../../source-img/in0.png",
                "../../source-img/in1.png",
                "../../source-img/in2.png",
                "../../source-img/in3.png",
                "../../source-img/in4.png",
                "../../source-img/in5.png",
                "../../source-img/in6.png",
                "../../source-img/in7.png",
                "../../source-img/in8.png",
                "../../source-img/in9.png");

    normalize_images(matrixL00double);
    normalize_images(matrixL01double);
    normalize_images(matrixL02double);
    normalize_images(matrixL03double);
    normalize_images(matrixL04double);
    normalize_images(matrixL05double);
    normalize_images(matrixL06double);
    normalize_images(matrixL07double);
    normalize_images(matrixL08double);
    normalize_images(matrixL09double);

}

static void create_layers_ocs() {
    ocl.create_layers();
}

static void reset_csc() {
    for (int i=0; i < 37; i++) {
        csc[i] = 0;
    }
    ocl.zero_CSC();
}

static void write_layers() {

    ocl.create_layers();
    ocl.write_weights(matrixW11double, matrixW12double,
                      matrixW21double, matrixW22double,
                      matrixW31double, matrixW32double, matrixW33double,
                      matrixW41double, matrixW42double, matrixW43double,
                      matrixW51double, matrixW52double, matrixW53double,
                      matrixW61double, matrixW62double, matrixW63double);
    ocl.write_bias(matrixB11double, matrixB12double,
                   matrixB21double, matrixB22double,
                   matrixB31double, matrixB32double, matrixB33double,
                   matrixB41double, matrixB42double, matrixB43double,
                   matrixB51double, matrixB52double, matrixB53double,
                   matrixB61double, matrixB62double, matrixB63double);

    ocl.create_bufs_abft(csc);
    ocl.write_weight_sums( matrixW11sum, matrixW12sum,
                           matrixW21sum, matrixW22sum,
                           matrixW31sum, matrixW32sum, matrixW33sum,
                           matrixW41sum, matrixW42sum, matrixW43sum,
                           matrixW51sum, matrixW52sum, matrixW53sum,
                           matrixW61sum, matrixW62sum, matrixW63sum,
                           matrixB11sum, matrixB12sum,
                           matrixB21sum, matrixB22sum,
                           matrixB31sum, matrixB32sum, matrixB33sum,
                           matrixB41sum, matrixB42sum, matrixB43sum,
                           matrixB51sum, matrixB52sum, matrixB53sum,
                           matrixB61sum, matrixB62sum, matrixB63sum);
}

static void check_layer() {
    //temp code for checking results
    double* inM;
    double* outR;
    double* outW;
    double* outB;

    inM = (double*)malloc(4096 * sizeof(double));
    outR = (double*)malloc(10 * sizeof(double));
    outW = (double*)malloc((c1d * c1d) * (c1w * c1h) * sizeof(double));
    outB = (double*)malloc((c1d) * sizeof(double));

    ocl.buf_read(1, 1, 4096, inM, ocl.ficBuf);
    ocl.buf_read(1, 1, 10, outR, ocl.fisBuf);

    /*for (int i = 0; i < 1; i++) {
        for (int j = 0; j < c1h; j++) {
            for (int k = 0; k < c1w; k++) {
                printf("%d, %d, %d: %f ", i, j, k, outR[(i * c1h * c1w) + (j * c1w) + k]);
            }
            printf("\n");
        }
    }*/
    for (int k = 0; k < 30; k++) {
        printf("%d: %f ", k, inM[k]);
    }
    printf("\n");

    for (int k = 0; k < 10; k++) {
        printf("%d: %f ", k, outR[k]);
    }
    printf("\n");


    /*for (int i = 0; i < c3d; i++) {
        for (int j = 0; j < c4h; j++) {
            for (int k = 0; k < c4w; k++) {
                if (outR[(i * c4h * c4w) + (j * c4w) + k] > 1000) {
                    printf("error! %d, %d, %d: %f ", i, j, k, outR[(i * c4h * c4w) + (j * c4w) + k]);
                    printf("\n");
                }
            }
        }
    }*/

    free(inM);
    free(outR);
    free(outW);
    free(outB);
}

static void forward() {
    //conv block 1
    //convolution 1-1
    ocl.convolution3(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c12Buf,
                          c1w, c1h, c10d, k1, c1pad, c1w, c1h, c1d);

    //check_layer();
    ocl.relu(ocl.c12Buf, ocl.c11Buf, c1w, c1h, c1d);
    //check_layer();

    //convolution 1-2
    ocl.convolution3(ocl.c11Buf, ocl.w12Buffer, ocl.b12Buffer, ocl.c12Buf,
                          c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d);

    ocl.relu(ocl.c12Buf, ocl.c11Buf, c1w, c1h, c1d);

    //max pool 1
    ocl.maxpool(ocl.c11Buf, ocl.c21Buf, c1w, c1h, c1d, 2, 2, c2w, c2h);


    //conv block 2
    //convolution 2-1
    ocl.convolution3(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                          c2w, c2h, c20d, k2, c2pad, c2w, c2h, c2d);
    ocl.relu(ocl.c22Buf, ocl.c21Buf, c2w, c2h, c2d);

    //convolution 2-2 / 22 -> 21
    ocl.convolution3(ocl.c21Buf, ocl.w22Buffer, ocl.b22Buffer, ocl.c22Buf,
                          c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d);
    ocl.relu(ocl.c22Buf, ocl.c21Buf, c2w, c2h, c2d);

    //max pool 2
    ocl.maxpool(ocl.c21Buf, ocl.c31Buf, c2w, c2h, c2d, 2, 2, c3w, c3h);


    //conv block 3
    //convolution 3-1
    ocl.convolution3(ocl.c31Buf, ocl.w31Buffer, ocl.b31Buffer, ocl.c32Buf,
                          c3w, c3h, c30d, k3, c3pad, c3w, c3h, c3d);
    ocl.relu(ocl.c32Buf, ocl.c31Buf, c3w, c3h, c3d);

    //convolution 3-2
    ocl.convolution3(ocl.c31Buf, ocl.w32Buffer, ocl.b32Buffer, ocl.c32Buf,
                          c3w, c3h, c3d, k3, c3pad, c3w, c3h, c3d);
    ocl.relu(ocl.c32Buf, ocl.c31Buf, c3w, c3h, c3d);

    //convolution 3-3
    ocl.convolution3(ocl.c31Buf, ocl.w33Buffer, ocl.b33Buffer, ocl.c32Buf,
                          c3w, c3h, c3d, k3, c3pad, c3w, c3h, c3d);
    ocl.relu(ocl.c32Buf, ocl.c31Buf, c3w, c3h, c3d);

    //max pool 3
    ocl.maxpool(ocl.c31Buf, ocl.c41Buf, c3w, c3h, c3d, 2, 2, c4w, c4h);


    //conv block 4
    //convolution 4-1
    ocl.convolution3(ocl.c41Buf, ocl.w41Buffer, ocl.b41Buffer, ocl.c42Buf,
                          c4w, c4h, c40d, k4, c4pad, c4w, c4h, c4d); // abft trigger after reboot
    ocl.relu(ocl.c42Buf, ocl.c41Buf, c4w, c4h, c4d);

    //convolution 4-2
    ocl.convolution3(ocl.c41Buf, ocl.w42Buffer, ocl.b42Buffer, ocl.c42Buf,
                          c4w, c4h, c4d, k4, c4pad, c4w, c4h, c4d);
    ocl.relu(ocl.c42Buf, ocl.c41Buf, c4w, c4h, c4d);

    //convolution 4-3
    ocl.convolution3(ocl.c41Buf, ocl.w43Buffer, ocl.b43Buffer, ocl.c42Buf,
                          c4w, c4h, c4d, k4, c4pad, c4w, c4h, c4d);
    ocl.relu(ocl.c42Buf, ocl.c41Buf, c4w, c4h, c4d);

    //max pool 4
    ocl.maxpool(ocl.c41Buf, ocl.c51Buf, c4w, c4h, c4d, 2, 2, c5w, c5h);


    //conv block 5
    //convolution 5-1
    ocl.convolution3(ocl.c51Buf, ocl.w51Buffer, ocl.b51Buffer, ocl.c52Buf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d);
    ocl.relu(ocl.c52Buf, ocl.c51Buf, c5w, c5h, c5d);

    //convolution 5-2
    ocl.convolution3(ocl.c51Buf, ocl.w52Buffer, ocl.b52Buffer, ocl.c52Buf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d);
    ocl.relu(ocl.c52Buf, ocl.c51Buf, c5w, c5h, c5d);

    //convolution 5-3
    ocl.convolution3(ocl.c51Buf, ocl.w53Buffer, ocl.b53Buffer, ocl.c52Buf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d);
    ocl.relu(ocl.c52Buf, ocl.c51Buf, c5w, c5h, c5d);

    //max pool 5
    ocl.maxpool(ocl.c51Buf, ocl.c61Buf, c5w, c5h, c5d, 2, 2, c6w, c6h);


    //mat block
    //matmul 6-1
    ocl.flatmat(ocl.c61Buf, ocl.w61Buffer, ocl.b61Buffer, ocl.c62Buf,
                     25088, 4096);
    ocl.relu(ocl.c62Buf, ocl.c63Buf, 4096, 1, 1);

    //matmul 6-2
    ocl.flatmat(ocl.c63Buf, ocl.w62Buffer, ocl.b62Buffer, ocl.c62Buf,
                     4096, 4096);
    ocl.relu(ocl.c62Buf, ocl.c63Buf, 4096, 1, 1);

    //matmul 6-3
    ocl.flatmat(ocl.c63Buf, ocl.w63Buffer, ocl.b63Buffer, ocl.c62Buf,
                     4096, 1000);
    ocl.relu(ocl.c62Buf, ocl.c6rBuf, 1000, 1, 1);
}

int forward_abft() {
    int abftflag = 0;

    //conv block 1
    //convolution 1-1
    ocl.convolution3_abft(ocl.l0Buffer, ocl.w11Buffer, ocl.b11Buffer, ocl.c12Buf,
                     ocl.icsBuf, ocl.w11sBuffer, ocl.b11sBuffer, ocl.ocsBuf,
                     c1w, c1h, c10d, k1, c1pad, c1w, c1h, c1d, 0);

    ocl.relu_dmr(ocl.c12Buf, ocl.c1dBuf, ocl.c11Buf, c1w, c1h, c1d, 1);

    //convolution 1-2
    ocl.convolution3_abft(ocl.c11Buf, ocl.w12Buffer, ocl.b12Buffer, ocl.c12Buf,
                          ocl.icsBuf, ocl.w12sBuffer, ocl.b12sBuffer, ocl.ocsBuf,
                          c1w, c1h, c1d, k1, c1pad, c1w, c1h, c1d, 2);

    ocl.relu_dmr(ocl.c12Buf, ocl.c1dBuf, ocl.c11Buf, c1w, c1h, c1d, 3);

    //max pool 1
    ocl.maxpool_dmr(ocl.c11Buf, ocl.c2dBuf, ocl.c21Buf, c1w, c1h, c1d, 2, 2, c2w, c2h, 4);


    //conv block 2
    //convolution 2-1
    ocl.convolution3_abft(ocl.c21Buf, ocl.w21Buffer, ocl.b21Buffer, ocl.c22Buf,
                          ocl.icsBuf, ocl.w21sBuffer, ocl.b21sBuffer, ocl.ocsBuf,
                          c2w, c2h, c20d, k2, c2pad, c2w, c2h, c2d, 5);
    ocl.relu_dmr(ocl.c22Buf, ocl.c2dBuf, ocl.c21Buf, c2w, c2h, c2d, 6);

    //convolution 2-2 / 22 -> 21
    ocl.convolution3_abft(ocl.c21Buf, ocl.w22Buffer, ocl.b22Buffer, ocl.c22Buf,
                          ocl.icsBuf, ocl.w22sBuffer, ocl.b22sBuffer, ocl.ocsBuf,
                          c2w, c2h, c2d, k2, c2pad, c2w, c2h, c2d, 7);
    ocl.relu_dmr(ocl.c22Buf, ocl.c2dBuf, ocl.c21Buf, c2w, c2h, c2d, 8);

    //max pool 2
    ocl.maxpool_dmr(ocl.c21Buf, ocl.c3dBuf, ocl.c31Buf, c2w, c2h, c2d, 2, 2, c3w, c3h, 9);


    //conv block 3
    //convolution 3-1
    ocl.convolution3_abft(ocl.c31Buf, ocl.w31Buffer, ocl.b31Buffer, ocl.c32Buf,
                          ocl.icsBuf, ocl.w31sBuffer, ocl.b31sBuffer, ocl.ocsBuf,
                          c3w, c3h, c30d, k3, c3pad, c3w, c3h, c3d, 10);
    ocl.relu_dmr(ocl.c32Buf, ocl.c3dBuf, ocl.c31Buf, c3w, c3h, c3d, 11);

    //convolution 3-2
    ocl.convolution3_abft(ocl.c31Buf, ocl.w32Buffer, ocl.b32Buffer, ocl.c32Buf,
                          ocl.icsBuf, ocl.w32sBuffer, ocl.b32sBuffer, ocl.ocsBuf,
                          c3w, c3h, c3d, k3, c3pad, c3w, c3h, c3d, 12);
    ocl.relu_dmr(ocl.c32Buf, ocl.c3dBuf, ocl.c31Buf, c3w, c3h, c3d, 13);

    //convolution 3-3
    ocl.convolution3_abft(ocl.c31Buf, ocl.w33Buffer, ocl.b33Buffer, ocl.c32Buf,
                          ocl.icsBuf, ocl.w33sBuffer, ocl.b33sBuffer, ocl.ocsBuf,
                          c3w, c3h, c3d, k3, c3pad, c3w, c3h, c3d, 14);
    ocl.relu_dmr(ocl.c32Buf, ocl.c3dBuf, ocl.c31Buf, c3w, c3h, c3d, 15);

    //max pool 3
    ocl.maxpool_dmr(ocl.c31Buf, ocl.c4dBuf, ocl.c41Buf, c3w, c3h, c3d, 2, 2, c4w, c4h, 16);


    //conv block 4
    //convolution 4-1
    ocl.convolution3_abft_icr(ocl.c41Buf, ocl.w41Buffer, ocl.b41Buffer, ocl.c42Buf,
                              ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf,  ocl.w41sBuffer, ocl.b41sBuffer, ocl.ocsBuf,
                          c4w, c4h, c40d, k4, c4pad, c4w, c4h, c4d, 17); // abft trigger after reboot
    ocl.relu_dmr(ocl.c42Buf, ocl.c4dBuf, ocl.c41Buf, c4w, c4h, c4d, 18);

    //convolution 4-2
    ocl.convolution3_abft_icr(ocl.c41Buf, ocl.w42Buffer, ocl.b42Buffer, ocl.c42Buf,
                              ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf,  ocl.w42sBuffer, ocl.b42sBuffer, ocl.ocsBuf,
                          c4w, c4h, c4d, k4, c4pad, c4w, c4h, c4d, 19);
    ocl.relu_dmr(ocl.c42Buf, ocl.c4dBuf, ocl.c41Buf, c4w, c4h, c4d, 20);

    //convolution 4-3
    ocl.convolution3_abft_icr(ocl.c41Buf, ocl.w43Buffer, ocl.b43Buffer, ocl.c42Buf,
                              ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf,  ocl.w43sBuffer, ocl.b43sBuffer, ocl.ocsBuf,
                          c4w, c4h, c4d, k4, c4pad, c4w, c4h, c4d, 21);
    ocl.relu_dmr(ocl.c42Buf, ocl.c4dBuf, ocl.c41Buf, c4w, c4h, c4d, 22);

    //max pool 4
    ocl.maxpool_dmr(ocl.c41Buf, ocl.c5dBuf, ocl.c51Buf, c4w, c4h, c4d, 2, 2, c5w, c5h, 23);


    //conv block 5
    //convolution 5-1
    ocl.convolution3_abft_icr(ocl.c51Buf, ocl.w51Buffer, ocl.b51Buffer, ocl.c52Buf,
                          ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf, ocl.w51sBuffer, ocl.b51sBuffer, ocl.ocsBuf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d, 24);
    ocl.relu_dmr(ocl.c52Buf, ocl.c5dBuf, ocl.c51Buf, c5w, c5h, c5d, 25);

    //convolution 5-2
    ocl.convolution3_abft_icr(ocl.c51Buf, ocl.w52Buffer, ocl.b52Buffer, ocl.c52Buf,
                              ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf,  ocl.w52sBuffer, ocl.b52sBuffer, ocl.ocsBuf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d, 26);
    ocl.relu_dmr(ocl.c52Buf, ocl.c5dBuf, ocl.c51Buf, c5w, c5h, c5d, 27);

    //convolution 5-3
    ocl.convolution3_abft_icr(ocl.c51Buf, ocl.w53Buffer, ocl.b53Buffer, ocl.c52Buf,
                              ocl.cicBuf, ocl.cisrBuf, ocl.cisBuf,  ocl.w53sBuffer, ocl.b53sBuffer, ocl.ocsBuf,
                          c5w, c5h, c5d, k5, c5pad, c5w, c5h, c5d, 28);
    ocl.relu_dmr(ocl.c52Buf, ocl.c5dBuf, ocl.c51Buf, c5w, c5h, c5d, 29);

    //max pool 5
    ocl.maxpool_dmr(ocl.c51Buf, ocl.c5dBuf, ocl.c61Buf, c5w, c5h, c5d, 2, 2, c6w, c6h, 30);


    //mat block
    //matmul 6-1
    ocl.flatmat_abft(ocl.c61Buf, ocl.w61Buffer, ocl.b61Buffer, ocl.c62Buf,
                          ocl.ficBuf, ocl.fisBuf, ocl.w61sBuffer, ocl.b61sBuffer, ocl. ocsBuf,
                          25088, 4096, 31);
    ocl.relu_dmr(ocl.c62Buf, ocl.c6dBuf, ocl.c63Buf, 4096, 1, 1, 32);

    //check_layer();

    //matmul 6-2
    ocl.flatmat_abft(ocl.c63Buf, ocl.w62Buffer, ocl.b62Buffer, ocl.c62Buf,
                     ocl.ficBuf, ocl.fisBuf, ocl.w62sBuffer, ocl.b62sBuffer, ocl. ocsBuf,
                          4096, 4096, 33);
    ocl.relu_dmr(ocl.c62Buf, ocl.c6dBuf, ocl.c63Buf, 4096, 1, 1, 34);

    //matmul 6-3
    ocl.flatmat_abft(ocl.c63Buf, ocl.w63Buffer, ocl.b63Buffer, ocl.c62Buf,
                     ocl.ficBuf, ocl.fisBuf, ocl.w63sBuffer, ocl.b63sBuffer, ocl. ocsBuf,
                          4096, 1000,  35);
    ocl.relu_dmr(ocl.c62Buf, ocl.c6dBuf, ocl.c6rBuf, 1000, 1, 1, 36);

    return abftflag;
}

int predictImage(double* input, double* output) {
    double max = 0;
    int maxind;

    //input 0
    ocl.write_image(input);
    forward();
    ocl.buf_read(1, 1, 1000, output, ocl.c6rBuf);
    //select max output value for non-abft

    for (int i=0; i<1000;i++) {
        //printf("%f ", matrixR6[(i * 10) + j]);
        if (output[i] > max) {
            max = output[i];
            maxind = i;
        }
    }

    return maxind;
}

int predictImage_abft(double* input, double* output) {
    int abfttrigger = 0;
    double max = 0;
    int maxind;

    abft_error_flag = 0;

    ocl.write_image(input);
    forward_abft();
    ocl.buf_read(1, 1, 1000, output, ocl.c6rBuf);
    ocl.buf_read(1, 1, 37, csc, ocl.cscBuf);
    //select max output value for abft

    for (int i=0; i<1000;i++) {
        if (output[i] > max) {
            max = output[i];
            maxind = i;
        }
    }

    for (int i=0; i<37;i++) {
        if (fabs(csc[i]) > 0.1) {
            abfttrigger = 1;
            total_abft_errors++; //can be disabled for overhead calcs
        }
    }
    if (abfttrigger == 1) {
        abft_error_flag = 1;
        printf("ABFT flag triggered! \n");
        for (int i=0; i<37;i++) {
            printf("%f ", csc[i]);
        }
        printf("\n");
        total_abft_inference_errors++;
        abfttrigger = 0;
    }

    reset_csc();

    return maxind;
}


struct PredictImages : public IProgram {
    int run() override {
        int maxind[10];

        maxind[0] = predictImage(matrixL00double, matrixResult0);
        maxind[1] = predictImage(matrixL01double, matrixResult1);
        maxind[2] = predictImage(matrixL02double, matrixResult2);
        maxind[3] = predictImage(matrixL03double, matrixResult3);
        maxind[4] = predictImage(matrixL04double, matrixResult4);
        maxind[5] = predictImage(matrixL05double, matrixResult5);
        maxind[6] = predictImage(matrixL06double, matrixResult6);
        maxind[7] = predictImage(matrixL07double, matrixResult7);
        maxind[8] = predictImage(matrixL08double, matrixResult8);
        maxind[9] = predictImage(matrixL09double, matrixResult9);

        printf("predictions: \n");
        for (int i =0; i <10; i++) {
            printf("%d ", maxind[i]);
        }
        printf("\n");

        return 1;
    }
};

struct PredictImages_abft : public IProgram {
    int run() override {
        int maxind[10];

        maxind[0] = predictImage_abft(matrixL00double, matrixResult0);
        maxind[1] = predictImage_abft(matrixL01double, matrixResult1);
        maxind[2] = predictImage_abft(matrixL02double, matrixResult2);
        maxind[3] = predictImage_abft(matrixL03double, matrixResult3);
        maxind[4] = predictImage_abft(matrixL04double, matrixResult4);
        maxind[5] = predictImage_abft(matrixL05double, matrixResult5);
        maxind[6] = predictImage_abft(matrixL06double, matrixResult6);
        maxind[7] = predictImage_abft(matrixL07double, matrixResult7);
        maxind[8] = predictImage_abft(matrixL08double, matrixResult8);
        maxind[9] = predictImage_abft(matrixL09double, matrixResult9);

        printf("predictions: \n");
        for (int i =0; i <10; i++) {
            ref_prediction[i] = maxind[i]; //todo remove this
            printf("%d ", maxind[i]);
        }
        printf("\n");

        return 1;
    }
};

struct PredictImages_ec : public IProgram {
    int run() override {
        int outputError = 0;
        int sigOutputError = 0;
        int totalError = 0;
        int maxind[10];

        double diffVal = 0.000001;

        maxind[0] = predictImage_abft(matrixL00double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult0[i]) {
                //printf("Input 0 output error! o: %f, ref:%f \n", matrixResult0[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult0[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 0 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[1] = predictImage_abft(matrixL01double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult1[i]) {
                //printf("Input 1 output error! o: %f, ref:%f \n", matrixResult1[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult1[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 1 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[2] = predictImage_abft(matrixL02double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult2[i]) {
                //printf("Input 2 output error! o: %f, ref:%f \n", matrixResult2[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult2[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 2 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[3] = predictImage_abft(matrixL03double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult3[i]) {
                //printf("Input 3 output error! o: %f, ref:%f \n", matrixResult3[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult3[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 3 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[4] = predictImage_abft(matrixL04double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult4[i]) {
                //printf("Input 4 output error! o: %f, ref:%f \n", matrixResult4[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult4[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 4 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[5] = predictImage_abft(matrixL05double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult5[i]) {
                //printf("Input 5 output error! o: %f, ref:%f \n", matrixResult5[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult5[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 5 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[6] = predictImage_abft(matrixL06double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult6[i]) {
                //printf("Input 6 output error! o: %f, ref:%f \n", matrixResult6[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult6[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 6 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[7] = predictImage_abft(matrixL07double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult7[i]) {
                //printf("Input 7 output error! o: %f, ref:%f \n", matrixResult7[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult7[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 7 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[8] = predictImage_abft(matrixL08double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult8[i]) {
                //printf("Input 8 output error! o: %f, ref:%f \n", matrixResult8[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult8[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 8 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
        }
        sigOutputError = 0;
        maxind[9] = predictImage_abft(matrixL09double, matrixR6);
        for (int i=0; i<1000;i++) {
            if (matrixR6[i] != matrixResult9[i]) {
                //printf("Input 9 output error! o: %f, ref:%f \n", matrixResult9[i], matrixR6[i]);
                outputError++;
                totalError++;
                if (fabs(matrixR6[i] - matrixResult9[i]) > diffVal) {
                    sigOutputError++;
                    total_sig_output_errors++;
                }
            }
        }
        if (outputError > 0) {
            printf("Input 9 output error! Error count: %d \n", outputError);
            outputError = 0;
            total_sig_layer_output_errors++;
            if (abft_error_flag == 0) {
                total_false_negatives++;
            }
        }
        if ((sigOutputError > 0) && (abft_error_flag == 0)) {
            total_sig_false_negatives++;
            sigOutputError = 0;
        }


        if (totalError > 0) {
            printf("Total error count: %d \n", totalError);
            total_output_errors += totalError;
            totalError = 0;
        }

        for (int i =0; i <10; i++) {
            if (maxind[i] != ref_prediction[i]) {
                printf("Prediction error! p: %d, ref:%d \n", maxind[i], ref_prediction[i]);
                total_prediction_error++;
            }
        }

        return 1;
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
    PredictImages predictImages;
    PredictImages_abft predictImagesAbft;
    PredictImages_ec predictImages_ec;

    int result = 0;

    createVectors();
    copyModel();
    copyWeightSums();
    //matrixW41sum[4] = 0.4;
    //create_layers_ocs();
    write_layers();

    load_inputs();

    double time1 = 0;
    double time2 = 0;
    double time3 = 0;
    //checking non-abft runtime
    for (int i= 0; i < 10; i++) {
        result = Program_sw.runProgram(predictImages);
        time1 += Program_sw.getElapsedTime();
    }
    std::cout << "prediction: " << result << std::endl;
    std::cout << "Elapsed time: " << time1 << " us" << std::endl;

    //checking abft overhead
    for (int i= 0; i < 10; i++) {
        result = Program_sw.runProgram(predictImagesAbft);
        time2 += Program_sw.getElapsedTime();
    }
    std::cout << "prediction with abft: " << result << std::endl;
    std::cout << "Elapsed time: " << time2 << " us" << std::endl;
    std::cout << "Time difference: " << time2 - time1 << " us" << std::endl;
    std::cout << "abft overhead: " << ((time2 - time1) / time1) << " " << std::endl<< std::endl;


    //error-checking part
    //save_result();

    load_result();

    //Prediction with all error detection
    for (int i= 0; i < 20; i++) {
        //createVectors();        

        //copyModel();
        //copyWeightSums();

        //write_layers();

        //load_inputs();

        result = Program_sw.runProgram(predictImages_ec);
        time3 += Program_sw.getElapsedTime();
        //ocl.free_bufs();

        //freememory();
	
	/*
        printf("Total ABFT error count: %d \n", total_abft_errors);
        printf("Total output error count: %d \n", total_output_errors);
        printf("Total significant output error count: %d \n", total_sig_output_errors);
        printf("Total significant output error layers count: %d \n", total_sig_layer_output_errors);
        printf("Total prediction error count: %d \n\n", total_prediction_error);
        */

        printf("i: %d \n", i);
        printf("Total ABFT inference error count: %d \n", total_abft_inference_errors);

        printf("Total false negatives count: %d \n", total_false_negatives);
        printf("Total significant false negatives count: %d \n", total_sig_false_negatives);
        printf("Total prediction error count: %d \n", total_prediction_error);
    }
    std::cout << "prediction with ec: " << result << std::endl;
    std::cout << "Elapsed time: " << time3 << " us" << std::endl;

    sw.saveEndPoint();
    std::cout << "Total elapsed time: " << sw.getElapsedTime() << " us\n" << std::endl;

    std::cout << "Run-times: " << std::endl;
    std::cout << "Prediction: " << time1 << " us" << std::endl;
    std::cout << "Prediction with abft: " << time2 << " us" << std::endl;
    std::cout << "Prediction with EC: " << time2 << " us" << std::endl << std::endl;

    printf("Total ABFT error count: %d \n", total_abft_errors);
    printf("Total ABFT inference error count: %d \n\n", total_abft_inference_errors);
    printf("Total output error count: %d \n", total_output_errors);
    printf("Total significant output error count: %d \n\n", total_sig_output_errors);
    printf("Total inference output error count: %d \n", total_sig_layer_output_errors);
    printf("Total significant inference output error count: %d \n\n", total_sig_layer_output_errors);
    printf("Total false negatives count: %d \n", total_false_negatives);
    printf("Total significant false negatives count: %d \n\n", total_sig_false_negatives);
    printf("Total prediction error count: %d \n\n", total_prediction_error);

    //cleaning bufs and memory allocation
    freememory();

    //print opencl information
    printPlatformInfo(false);

    return 0;
}
