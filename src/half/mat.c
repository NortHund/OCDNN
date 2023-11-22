/**
1.
Fixed Point Implementation of N*M in M*K matrix Multiplication (2 point)
This program is just a nested loop. The only difference is that it should be implemented
as a fixed point program, i.e. it’s inputs and outputs are all FLOAT32 but the middle
computations all will be fixed point and half floating point and the final results again
must be FLOAT32 either printed in screen or stored in a float32 array. Please use the
following function format so I can use it in my test code to check the results:
int Matrxi_Mul( float * A, float * B, float *Result, int rows_of_A, int columns_of_A ,int rows_of_B, int
columns_of_B, bool Fix_Or_Float16)
// A is first matrix, B is the second, Result is the resultant matrix which obviously would be of size
rows_of_A in
//Fix_Or_Float16 determines if the middle computations are fixed or float 16
//
return 1 ; //if multiplication can be done
//
return 0 ; //if multiplication cannot be done
**/


//#include "file_half.h"
//#include "half_float.h"
//#include "half_float.c"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "half_float.h"
#include "half_float.c"
#include "half_ops.h"
#include "half_ops.c"


int Matrxi_Mul( float * A, float * B, float * Result, int rows_of_A, int columns_of_A ,int rows_of_B, int columns_of_B, bool Fix_Or_Float16) {
    int calculatable = 0;
    int index = 0;
    float matA[rows_of_A][columns_of_A];
    float matB[rows_of_B][rows_of_B];
    float matR[rows_of_A][columns_of_B];
    
    if (columns_of_A != rows_of_B) {
        calculatable = 1;
        return 0;
    }
    
    for(int i=0;i<rows_of_A;i++) {
        for(int j=0;j<columns_of_A;j++) {
            matA[i][j] = A[index];
            index++;
        }
    }
    
    index = 0;
    for(int i=0;i<rows_of_B;i++) {
        for(int j=0;j<columns_of_B;j++) {
            matB[i][j] = B[index];
            index++;
        }
    }
    
    printf("Matrix A\n");
    for(int i=0;i<rows_of_A;i++) {
        for(int j=0;j<columns_of_B;j++) {
            printf("%.2f ", matA[i][j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Matrix B\n");
    for(int i=0;i<rows_of_A;i++) {
        for(int j=0;j<columns_of_B;j++) {
            printf("%.2f ", matB[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    float adder = 0;
    float matAvalue = 0;
    float matBvalue = 0;
    float matRvalue = 0;
    
    if (Fix_Or_Float16 == true) { //calculations in HALF
        HALF halfAdder;
        HALF matAhalf;
        HALF matBhalf;
        HALF matRhalf;

        for(int i=0;i<columns_of_A;i++) {
            for(int j=0;j<rows_of_B;j++) {
                adder = 0;
                matRvalue = 0;
                matRhalf = floatToHALF(0);
                halfAdder = floatToHALF(0);
                for (int k=0;k<rows_of_B;k++) {
                    matAvalue = matA[i][k];
                    matBvalue = matB[k][j];
                    
                    matAhalf = floatToHALF(matAvalue);
                    matBhalf = floatToHALF(matBvalue);
                    
                    halfAdder = mulh(matAhalf, matBhalf);
                    matRhalf = addh(matRhalf, halfAdder);
                }
                matRvalue = HALFToFloat(matRhalf);
                matR[i][j] = matRvalue;
            }
        }
        
    } else { //calculations in fixed point
    
        float adder = 0;
        float matAvalue = 0;
        float matBvalue = 0;
        float matRvalue = 0;

        for(int i=0;i<columns_of_A;i++) {
            for(int j=0;j<rows_of_B;j++) {
                adder = 0;
                matRvalue = 0;
                for (int k=0;k<rows_of_B;k++) {
                    matAvalue = matA[i][k];
                    matBvalue = matB[k][j];
                    
                    adder = matAvalue * matBvalue;
                    matRvalue = matRvalue + adder;
                }
                matR[i][j] = matRvalue;
            }
        }
        
    }
    
    printf("Result:\n");
    for(int i=0;i<rows_of_A;i++) {
        for(int j=0;j<columns_of_B;j++) {
            printf("%.4f ", matR[i][j]);
        }
        printf("\n");
    }
    
    index = 0;
    for(int i=0;i<rows_of_A;i++) {
        for(int j=0;j<columns_of_B;j++) {
            Result[index] = matR[i][j];
            index++;
        }
    }
    
    return 1;
}

int main (int argc, char *argv[])  {
    float matrixA[4] = {1, 2, 
                        3, 4};
    float matrixB[4] = {1, 2, 
                        3, 4};
    
    int Acols = 2;
    int Arows = 2; 
    int Bcols = 2;
    int Brows = 2;
     
    float resultMatrix[Arows * Bcols];  
    
    bool FoF = true;
    
    int res = Matrxi_Mul((float *)matrixA, (float *)matrixB, (float *)resultMatrix, Arows, Acols, Brows, Bcols, FoF);
    printf("function returns: %d\n", res);
    
    
}
