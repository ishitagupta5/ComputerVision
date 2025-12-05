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
        printf("Usage: ./sobel_video_export input.mp4\n");
        return 1;
    }

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        printf("Error: Cannot open input video\n");
        return 1;
    }

    int width  = cap.get(CAP_PROP_FRAME_WIDTH);
    int height = cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(CAP_PROP_FPS);

    if (fps < 1 || fps > 240) fps = 30; // fallback

    printf("Processing %s (%dx%d @ %.2f FPS)\n", argv[1], width, height, fps);

    // Output writer: MP4 (H264)
    VideoWriter writer(
        "sobel_output.mp4",
        cv::VideoWriter::fourcc('a','v','c','1'),  // H264 codec
        fps,
        Size(width, height),
        false // grayscale video
    );

    if (!writer.isOpened()) {
        printf("Error: Cannot open output writer!\n");
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

        writer.write(sobel_mat);
        frame_count++;

        if (frame_count % 50 == 0)
            printf("Processed %d frames...\n", frame_count);
    }

    printf("Done. Saved sobel_output.mp4 (%d frames).\n", frame_count);

    cap.release();
    writer.release();
    return 0;
}
