#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void sobel_filter(uint8_t *src_arr, uint8_t *dest_arr,
                  int height, int width, int threshold, int metric)
{
    int x_sum, y_sum;

    #pragma omp parallel for schedule(static) private(x_sum, y_sum)
    for (int x = 1; x < height - 1; x++)
    {
        for (int y = 1; y < width - 1; y++)
        {
            x_sum = (
                src_arr[(x + 1)*width + (y + 1)] -
                src_arr[(x + 1)*width + (y - 1)] +
                (src_arr[x*width + (y + 1)] << 1) -
                (src_arr[x*width + (y - 1)] << 1) +
                src_arr[(x - 1)*width + (y + 1)] -
                src_arr[(x - 1)*width + (y - 1)]
            );

            y_sum = (
                src_arr[(x + 1)*width + (y + 1)] +
                (src_arr[(x + 1)*width + y] << 1) +
                src_arr[(x + 1)*width + (y - 1)] -
                src_arr[(x - 1)*width + (y + 1)] -
                (src_arr[(x - 1)*width + y] << 1) -
                src_arr[(x - 1)*width + (y - 1)]
            );

            double distance;
            if (metric)
                distance = sqrt((double)x_sum * x_sum + (double)y_sum * y_sum);
            else
                distance = abs(x_sum) + abs(y_sum);

            dest_arr[x * width + y] = (distance > threshold ? 255 : 0);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./sobel_video input.mp4\n");
        return 1;
    }

    //for compatible cmaera i would replace it to cap(0)
    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        printf("Error: Cannot open video file\n");
        return 1;
    }

    int width  = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);

    printf("Processing video: %s (%dx%d)\n", argv[1], width, height);

    Mat frame, gray, sobel_mat(height, width, CV_8UC1);
    uint8_t *dest = sobel_mat.data;

    double prev_time = omp_get_wtime();

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        double start = omp_get_wtime();
        sobel_filter(gray.data, dest, height, width, 150, 1);
        double end = omp_get_wtime();

        double frame_time = end - start;
        double fps = 1.0 / (end - prev_time);
        prev_time = end;

        printf("Frame time: %.4f sec | FPS: %.2f\n", frame_time, fps);

        imshow("Parallel Sobel Output", sobel_mat);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
