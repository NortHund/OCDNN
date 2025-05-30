__kernel void convolution_double(__global double* input, __global double* weight, __global double* bias, __global double* output,
                                int id, int ih, int iw, int kwh, int pad) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int layer = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int depth = get_global_size(2);

    //padding done for left/up side, but not for right/down side?

    double sum = 0;
    for (int h = 0; h < id; h++) {
        for (int j = 0; j < kwh; j++) {
            for (int k = 0; k < kwh; k++) {
                //checking if location is within bounds
                if ((row + j - pad) >= 0 && (row + j - pad) < ih &&
                    (col + k - pad) >= 0 && (col + k - pad) < iw) {
                    sum += input[(h * ih * iw) + ((row + j - pad) * iw) + (col + k - pad)] * weight[(h * depth * kwh * kwh) + (layer * kwh * kwh) + (j * kwh) + k];
                }
            }
        }
    }

  output[(layer * width * height) + (row * width) + col] = sum + bias[layer];
  //output[(layer * width * height) + (row * width) + col] = 1;
  //output[(layer * width * height) + (row * width) + col] = 0;
  //output[(layer * width * height) + (row * width) + col] = depth;
  //output[(layer * width * height) + (row * width) + col] = weight[0];
}

__kernel void convolution_ic(__global double* input, __global double* weight, __global double* bias, __global double* output,
                                int id, int ih, int iw, int kwh, int pad) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int layer = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int depth = get_global_size(2);

    //padding done for left/up side, but not for right/down side?

    double sum = 0;
    for (int j = 0; j < kwh; j++) {
        for (int k = 0; k < kwh; k++) {
            //checking if location is within bounds
            if ((row + j - pad) >= 0 && (row + j - pad) < ih &&
                (col + k - pad) >= 0 && (col + k - pad) < iw) {
                sum += input[(layer * ih * iw) + ((row + j - pad) * iw) + (col + k - pad)] * weight[(layer * kwh * kwh) + (j * kwh) + k];
            }
        }
    }

    if (layer == 0) {
        sum += bias[layer];
    }

    output[(layer * width * height) + (row * width) + col] = sum;
}

__kernel void input_sum(__global double* input, __global double* output, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

    for (int i = 0; i < depth; i++) {
      sum += input[(i * height * width) + (row * width) + (col)];
    }

  output[(row * width) + col] = sum;
}

__kernel void output_r(__global double* input, __global double* output, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int layer = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int layers = get_global_size(2);

    double sum = 0;

    for (int i = 0; i < depth; i++) {
      sum += input[(((layer * depth) + i) * height * width) + (row * width) + (col)];
    }

    output[(layer * height * width) + (row * width) + col] = sum;
}

__kernel void output_sum(__global double* input, __global double* output, int depth) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    double sum = 0;

    for (int i = 0; i < depth; i++) {
      sum += input[(i * height * width) + (row * width) + (col)];
    }

  output[(row * width) + col] = sum;
}

__kernel void cs_compare(__global double* inputCs, __global double* outputCs, __global double* result, int csInd) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int layer = get_global_id(2);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int depth = get_global_size(2);

    int wgNum = get_group_id(0);

    double diff = 0;

    diff = fabs(inputCs[(layer * width * height) + (row * width) + col] - outputCs[(layer * width * height) + (row * width) + col]);

    //change this to a very low value below 14 decimals and some results will start failing
    if (diff > 0.0000000001) {
        result[csInd] = diff + 1;
    }
    //result[csInd] = diff + 1;



}