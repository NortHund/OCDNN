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

__kernel void maxpool(__global double* input, __global double* output, int depth, int ih, int iw, int k_s, int st) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double max = 0;
    double val = 0;

    for (int i = 0; i < k_s; ++i) {
        for (int j = 0; j < k_s; ++j) {
            val = input[(depth * ih * iw) + (((row * st) + i) * iw) + ((col * st) + j)];
            if (val > max) {
                max = val;
            }
        }
    }
    output[(depth * height * width) + (row * width) + col] = max;

}

__kernel void maxpool_ref(__global double* input, __global double* output, int depth, int ih, int iw) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    const int len0 = (ih / height);
    const int len1 = (iw / width);
    int x0 = 0;
    int x1 = 0;
    int ismax = 0;
    for (int l0 = 0; l0 < len0; ++l0) {
        for (int l1 = 0; l1 < len1; ++l1) {
            ismax = input[(depth * ih * iw) + ((row * len0 + l0) * iw) +
                                   ((col * len1) + l1)]
                    > input[(depth * ih * iw) + ((row * len0 + x0) * iw) +
                                     ((col * len1) + x1)];
            x0 += ismax * (l0 - x0);
            x1 += ismax * (l1 - x1);
        }
    }
    output[(depth * height * width) + (row * width) + col] = input[
            (depth * ih * iw) + ((row * len0 + x0) * width) + ((col * len1) + x1)];

}