#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>   

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

void sobel_filter_acc(uint8_t *src, uint8_t *dst, int width, int height, int threshold)
{
    int size = width * height;

    double start = omp_get_wtime();

    #pragma acc data copyin(src[0:size]) copyout(dst[0:size])
    {
        #pragma acc parallel loop collapse(2)
        for (int x = 1; x < height - 1; x++) {
            for (int y = 1; y < width - 1; y++) {

                int gx =
                    src[(x+1)*width + (y+1)] -
                    src[(x+1)*width + (y-1)] +
                    2 * src[x*width + (y+1)] -
                    2 * src[x*width + (y-1)] +
                    src[(x-1)*width + (y+1)] -
                    src[(x-1)*width + (y-1)];

                int gy =
                    src[(x+1)*width + (y+1)] +
                    2 * src[(x+1)*width + y] +
                    src[(x+1)*width + (y-1)] -
                    src[(x-1)*width + (y+1)] -
                    2 * src[(x-1)*width + y] -
                    src[(x-1)*width + (y-1)];

                float magnitude = fabsf(gx) + fabsf(gy);

                dst[x * width + y] = (magnitude > threshold) ? 255 : 0;
            }
        }
    }

    double end = omp_get_wtime();
    printf("[GPU] Sobel time: %f seconds\n", end - start);
}


int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s input.png output.png\n", argv[0]);
        return 1;
    }

    int width, height, bpp;

    uint8_t *img = stbi_load(argv[1], &width, &height, &bpp, 1);
    if (!img) {
        printf("ERROR: Could not load image.\n");
        return 1;
    }

    uint8_t *sobel_out = calloc(width * height, sizeof(uint8_t));

    printf("Running OpenACC GPU Sobel on %dx%d image...\n", width, height);

    sobel_filter_acc(img, sobel_out, width, height, 150);

    stbi_write_png(argv[2], width, height, 1, sobel_out, width);

    stbi_image_free(img);
    free(sobel_out);

    printf("Saved result to %s\n", argv[2]);
    return 0;
}
