#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"



void sobel_filter(uint8_t * src_arr,uint8_t * dest_arr,int height, int width,int threshold,int metric)
{	
	/* Variable Declaration */
	int x_sum, y_sum;
    #pragma omp parallel for schedule(static) private(x_sum, y_sum)
	for (int x = 1; x < height - 1; x++) {
		/* 
		 * Apply the Sobel Filter's kernel convolution
		 * on each pixel of a single row.
		 * Convolution matrices:
		 * X:
		 * -1  0  1
		 * -2  0  2
		 * -1  0  1
		 * Y:
		 * -1 -2 -1
		 *  0  0  0
		 *  1  2  1
		 * Convolve with X to get Gx and with Y to get Gy
		 * The final pixel value is the Eucledian norm of Gx and Gy
		 */
		for (int y = 1; y < width - 1; y++) {
			x_sum = (
				src_arr[(x + 1)*width + (y + 1)] -
				src_arr[(x + 1)*width + (y - 1)] +
				(src_arr[   (x)*width + (y + 1)] << 1) -
				(src_arr[   (x)*width + (y - 1)] << 1) +
				src_arr[(x - 1)*width + (y + 1)] -
				src_arr[(x - 1)*width + (y - 1)]
			);

			y_sum = (
				src_arr[ (x + 1)*width + (y + 1)] +
				(src_arr[(x + 1)*width + (y)    ] << 1) +
				src_arr[ (x + 1)*width + (y - 1)] -
				src_arr[ (x - 1)*width + (y + 1)] -
				(src_arr[(x - 1)*width + (y)    ] << 1) -
				src_arr[ (x - 1)*width + (y - 1)]
			);

			// Manhatan Distance is used instead of Eucledian to increase performance
			//if metric is 1, use eucliden. otherwise use manhatan
			double distance; //type defines the precision

			if (metric){
                distance = sqrt((double)x_sum * x_sum +
                                (double)y_sum * y_sum);
			}
			else{
				distance = abs(x_sum) + abs(y_sum);
			}

			if (distance > threshold){
					dest_arr[x * width + y] = 255;
			} else {
					dest_arr[x * width + y] = 0;
			}

		}
	}

}

int main(int argc, char * argv[]){
    
    int width, height, bpp;
	// grayscale image where each pixel is a uint8_t 
    uint8_t* gray_image = stbi_load(argv[1], &width, &height, &bpp, 1);

	// output buffer for image after sobel filter
    uint8_t *sobel = calloc(width * height, sizeof(uint8_t));

	double start = omp_get_wtime();
    sobel_filter(gray_image, sobel, height, width, 150, 1);
    double end = omp_get_wtime();
    double cpu_time_used = end - start;
    
	// write output and free image
    stbi_write_png(argv[2],width,height,1,sobel,width);
    stbi_image_free(gray_image);
	free(sobel);
    
    printf("sobel filter took %f seconds\n", cpu_time_used);
    return 0;
}

