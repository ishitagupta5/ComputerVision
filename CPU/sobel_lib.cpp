#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern "C" {

__attribute__((visibility("default")))
void sobel_filter(
    uint8_t* src_arr,
    uint8_t* dest_arr,
    int height,
    int width,
    int threshold,
    int metric
) {
    int x_sum, y_sum;

    #pragma omp parallel for schedule(static) private(x_sum, y_sum)
    for (int x = 1; x < height - 1; x++) {
        for (int y = 1; y < width - 1; y++) {

            x_sum =
                src_arr[(x+1)*width + (y+1)] -
                src_arr[(x+1)*width + (y-1)] +
                (src_arr[x*width + (y+1)] << 1) -
                (src_arr[x*width + (y-1)] << 1) +
                src_arr[(x-1)*width + (y+1)] -
                src_arr[(x-1)*width + (y-1)];

            y_sum =
                src_arr[(x+1)*width + (y+1)] +
                (src_arr[(x+1)*width + y] << 1) +
                src_arr[(x+1)*width + (y-1)] -
                src_arr[(x-1)*width + (y+1)] -
                (src_arr[(x-1)*width + y] << 1) -
                src_arr[(x-1)*width + (y-1)];

            double distance = metric
                ? sqrt(x_sum * x_sum + y_sum * y_sum)
                : abs(x_sum) + abs(y_sum);

            dest_arr[x*width + y] = (distance > threshold ? 255 : 0);
        }
    }
}

}
