__kernel void relu(__global double* input, __global double* output, __global double* bias, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double biased_value = 0;
    biased_value = input[(depth * height * width) + (row * width) + (col)] + bias[depth];

    if (biased_value > 0) {
        output[(depth * height * width) + (row * width) + col] = biased_value;
    } else {
        output[(depth * height * width) + (row * width) + col] = 0;
    }

}

__kernel void maxpool(__global double* input, __global double* output, int ih, int iw, int k_s, int st) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int layer = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int depth = get_global_size(2);

    double max = 0;
    double val = 0;

    for (int i = 0; i < k_s; ++i) {
        for (int j = 0; j < k_s; ++j) {
            val = input[(layer * ih * iw) + (((row * st) + i) * iw) + ((col * st) + j)];
            if (val > max) {
                max = val;
            }
        }
    }
    output[(layer * height * width) + (row * width) + col] = max;

}

__kernel void flatmat(__global double* input, __global double* output, __global double* weights, __global double* bias, int id, int ih, int iw) {
    int col = get_global_id(0);
    int width = get_global_size(0);

    double sum = 0;

    for (int x = 0; x < (id * ih * iw); ++x) {
        sum += input[x] * weights[x * (width) + col];
    }

    output[col] = sum + bias[col];

}

__kernel void flatmat_ics(__global double* input, __global double* output, __global double* weights, __global double* bias, int id, int ih, int iw, int ow) {
    int col = get_global_id(0);
    int width = get_global_size(0);

    double sum = 0;

    for (int x = 0; x < (id * ih * iw); ++x) {
        sum += input[x] * weights[x * (width) + col];
    }

    for (int x = 0; x < (ow); ++x) {
            sum += bias[x];
    }

    output[0] = sum;
    //output[0] = 4;

}