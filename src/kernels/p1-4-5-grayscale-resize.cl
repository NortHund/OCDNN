__kernel void resize(int scale, __global unsigned char* input, __global unsigned char* output) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);
    int x = row * width + col;
    int x2 = (row * scale) * width * scale + (col * scale);

    output[x] = input[x2];
    
}

__kernel void grayscale_rgba(__global unsigned char* input, __global unsigned char* output) {
    int x = get_global_id(0);
    unsigned char grayscale = input[x/4 * 4] * 0.2126 + 
                                input[x/4 * 4 + 1] * 0.7152 + 
                                input[x/4 * 4 + 2] * 0.0722;
    output[x/4] = grayscale;
}
