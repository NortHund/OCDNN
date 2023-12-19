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

__kernel void maxpool(__global double* input, __global double* output, int depth) {
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