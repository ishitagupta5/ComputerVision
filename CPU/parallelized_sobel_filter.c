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
		int x_sum, y_sum;
		#pragma omp parallel for schedule(static) private(x_sum, y_sum)
	for (int x = 1; x < height - 1; x++) {
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

			double distance; 

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
    uint8_t* gray_image = stbi_load(argv[1], &width, &height, &bpp, 1);

    uint8_t *sobel = calloc(width * height, sizeof(uint8_t));

	double start = omp_get_wtime();
    sobel_filter(gray_image, sobel, height, width, 150, 1);
    double end = omp_get_wtime();
    double cpu_time_used = end - start;
    
    stbi_write_png(argv[2],width,height,1,sobel,width);
    stbi_image_free(gray_image);
	free(sobel);
    
    printf("sobel filter took %f seconds\n", cpu_time_used);
    return 0;
}

