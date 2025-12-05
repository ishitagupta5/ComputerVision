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
    if (argc < 3) {
        printf("Usage: ./sobel_video_headless input.mp4 output.mp4\n");
        return 1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        printf("Error: cannot open input video.\n");
        return 1;
    }

    int width  = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    printf("Processing video (%dx%d @ %.2f FPS)...\n", width, height, fps);

    // Write processed frames into output.mp4
    VideoWriter writer(
        argv[2],
        VideoWriter::fourcc('m','p','4','v'),
        fps,
        Size(width, height),
        false   // <-- grayscale output
    );

    if (!writer.isOpened()) {
        printf("Error: could not open output file for writing.\n");
        return 1;
    }

    Mat frame, gray, sobel_mat(height, width, CV_8UC1);
    uint8_t *dest = sobel_mat.data;

    int frame_count = 0;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        sobel_filter(gray.data, dest, height, width, 150, 1);

        // Write processed frame
        writer.write(sobel_mat);

        if (frame_count % 30 == 0)
            printf("Processed %d frames...\n", frame_count);

        frame_count++;
    }

    cap.release();
    writer.release();

    printf("Finished. Output saved to: %s\n", argv[2]);
    return 0;
}
