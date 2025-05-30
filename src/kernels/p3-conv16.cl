__kernel void convolution_half(__global half* inputA, __global half* inputB, __global half* output, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    half sum = 0;

    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
          sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)] * inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
        }
    }

  output[(layerB * width * height) + (row * width) + col] = sum;
}

__kernel void convolution_fl(__global float* inputA, __global float* inputB, __global float* output, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    float sum = 0;

    for (int i = 0; i < heightB; i++) {
        for (int j = 0; j < widthB; j++) {
          sum += inputA[(layerA * heightA * widthA) + ((row + i) * widthA) + (col + j)] * inputB[(layerA * depthB * heightB * widthB) + (layerB * heightB * widthB) + (i * widthB) + j];
        }
    }

  output[(layerB * width * height) + (row * width) + col] = sum;
}

__kernel void convolution_optim_ics(__global half* inM, __global half* wM, __global half* ics,
                                    __global half* midRW, __global half* midCL, __global half* cornerMat, __global half* matSum,
                                    int ih, int iw, int id, int wh, int ww, int oh, int ow, int od) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

        int k = ww;
        int k1 = (ww-1);
        int k2 = (ww-2);

        half midSQ = 0;
        for (int n = 0; n < id; n++) {
            for (int i = k1; i < ih - k1; i++) {
                for (int j = k1; j < iw - k1; j++) {
                    midSQ += inM[i * iw + j];
                }
            }
        }

        //half midRW[k1 * 2];
        int ind= 0;
        for (int i = 0; i < k1; i++) {
            midRW[ind] = 0;
            for (int j = k1; j < iw - k1; j++) {
                midRW[ind] += inM[i * iw + j];
            }
            ind++;
        }
        for (int i = ih - k1; i < ih; i++) {
            midRW[ind] = 0;
            for (int j = k1; j < iw - k1; j++) {
                midRW[ind] += inM[i * iw + j];
            }
            ind++;
        }

        //half midCL[k1*2];
        ind= 0;
        for (int i = 0; i < k1; i++) {
            midCL[ind] = 0;
            for (int j = k1; j < iw - k1; j++) {
                midCL[ind] += inM[j * iw + i];
            }
            ind++;
        }
        for (int i = ih - k1; i < ih; i++) {
            midCL[ind] = 0;
            for (int j = k1; j < iw - k1; j++) {
                midCL[ind] += inM[j * iw + i];
            }
            ind++;
        }

        //half cornerMat[(k1 * 2) * (k1 * 2)];
        ind= 0;
        for (int i = 0; i < k1; i++) {
            for (int j = 0; j < k1; j++) {
                cornerMat[ind] = inM[j * iw + i];
                ind++;
            }
            for (int j = iw - k1; j < iw; j++) {
                cornerMat[ind] = inM[j * iw + i];
                ind++;
            }
        }
        for (int i = ih - k1; i < ih; i++) {
            for (int j = 0; j < k1; j++) {
                cornerMat[ind] = inM[j * iw + i];
                ind++;
            }
            for (int j = iw - k1; j < iw; j++) {
                cornerMat[ind] = inM[j * iw + i];
                ind++;
            }
        }

        //matsum loop version 1
        //half matSum[k * k];
        /*for (int ci = 0; ci < k; ci++) {
            for (int cj = 0; cj < k; cj++) {
                matSum[ci * k + cj] = 0;
                matSum[ci * k + cj] += midSQ;
                for (int i = ci; i < ci + k1; i++) {
                    matSum[ci * k + cj] += midRW[i];
                    matSum[ci * k + cj] += midCL[i];
                    for (int j = cj; j < cj + k1; j++) {
                        matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + j];
                    }
                }
            }
        }*/

        //matsum loop version 2
        //first value is calculated in full
        matSum[0] = 0;
        matSum[0] += midSQ;
        for (int i = 0; i < k1; i++) {
            matSum[0] += midRW[i];
            matSum[0] += midCL[i];
            for (int j = 0; j < k1; j++) {
                matSum[0] += cornerMat[i * (k1 * 2) + j];
            }
        }
        half prevSum;
        //second value onwards with this loop, value based on previous value
        for (int ci = 0; ci < k; ci++) {
            if (ci % 2 == 0) {
                for (int cj = 0; cj < k; cj++) {
                    if (cj + ci > 0) {
                        if (cj == 0) { //downwards shift, left edge
                            matSum[ci * k + cj] = prevSum;
                            matSum[ci * k + cj] -= midRW[ci - 1];
                            matSum[ci * k + cj] += midRW[ci + k2];
                            for (int j = cj; j < cj + k1; j++) {
                                matSum[ci * k + cj] -= cornerMat[(ci - 1) * (k1 * 2) + j];
                                matSum[ci * k + cj] += cornerMat[(ci + k2) * (k1 * 2) + j];
                            }
                        } else { //left to right
                            matSum[ci * k + cj] = prevSum;
                            matSum[ci * k + cj] -= midCL[cj - 1];
                            matSum[ci * k + cj] += midCL[cj + k2];
                            for (int i = ci; i < ci + k1; i++) {
                                matSum[ci * k + cj] -= cornerMat[i * (k1 * 2) + cj - 1];
                                matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + cj + k2];
                            }
                        }
                        prevSum = matSum[ci * k + cj];
                    } else {
                        prevSum = matSum[0];
                    }
                }
            } else {
                for (int cj = k1; cj >= 0; cj--) {
                    if (cj == k1) { //downwards shift, right edge
                        matSum[ci * k + cj] = prevSum;
                        matSum[ci * k + cj] -= midRW[ci - 1];
                        matSum[ci * k + cj] += midRW[ci + k2];
                        for (int j = cj; j > cj - k1; j--) {
                            matSum[ci * k + cj] -= cornerMat[(ci - 1) * (k1 * 2) + j];
                            matSum[ci * k + cj] += cornerMat[(ci + k2) * (k1 * 2) + j];
                        }
                    } else { //right to left
                        matSum[ci * k + cj] = prevSum;
                        matSum[ci * k + cj] += midCL[cj];
                        matSum[ci * k + cj] -= midCL[cj + k1];
                        for (int i = ci; i < ci + k1; i++) {
                            matSum[ci * k + cj] -= cornerMat[i * (k1 * 2) + cj + k1];
                            matSum[ci * k + cj] += cornerMat[i * (k1 * 2) + cj];
                        }
                    }
                    prevSum = matSum[ci * k + cj];
                }
            }

        }

        half wSum = 0;
        half xSum = 0;
        half checksum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                wSum = 0;
                for (int m = 0; m < od; ++m) {
                    wSum += wM[(m * wh * ww) + (i * ww) + j];
                }
                checksum += wSum * matSum[(i * ww) + j];
            }
        }

  ics[0] = checksum;
}

__kernel void convolution_half_ics(__global half* inputA, __global half* inputB, __global half* ics, int heightA, int widthA, int depthA, int heightB, int widthB, int depthB, int layerA, int layerB) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    half wSum = 0;
    half xSum = 0;
    half checksum = 0;
    for (int n = 0; n < depthA; ++n) {
        for (int i = 0; i < heightB; ++i) {
            for (int j = 0; j < widthB; ++j) {
                wSum = 0;
                for (int m = 0; m < depthB; ++m) {
                    wSum += inputB[(n * depthB * heightB * widthB) + (m * heightB * widthB) + (i * widthB) + j];
                }

                xSum = 0;
                for (int r = 0; r < height; ++r) {
                    for (int c = 0; c < width; ++c) {
                        xSum += inputA[(n * heightA * widthA) + ((r + i) * widthA) + (c + j)];
                    }
                }
                checksum += xSum * wSum;
            }
        }
    }

  ics[0] = checksum;
}

__kernel void convolution_half_ocs(__global half* inputA, __global half* ocs, int heightA, int widthA, int depthA, int ind) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int width = get_global_size(0);
    int height = get_global_size(1);

    half sum = 0;

        for (int m = 0; m < depthA; m++) {
            for (int i = 0; i < heightA; i++) {
                for (int j = 0; j < widthA; j++) {
                  sum += inputA[(m * heightA * widthA) + (i * widthA) + j];
                }
            }
        }

  ocs[ind] = sum;
}