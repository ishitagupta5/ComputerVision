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

struct CpuTimes {
    unsigned long long user;
    unsigned long long nice;
    unsigned long long system;
    unsigned long long idle;
    unsigned long long iowait;
    unsigned long long irq;
    unsigned long long softirq;
    unsigned long long steal;
    unsigned long long guest;
    unsigned long long guest_nice;
};

bool read_cpu_times(CpuTimes &t) {
    FILE *fp = fopen("/proc/stat", "r");
    if (!fp) return false;

    int scanned = fscanf(fp, "cpu  %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu",
                         &t.user, &t.nice, &t.system, &t.idle, &t.iowait,
                         &t.irq, &t.softirq, &t.steal, &t.guest, &t.guest_nice);
    fclose(fp);
    return (scanned >= 4);
}

double compute_cpu_usage(const CpuTimes &start, const CpuTimes &end) {
    unsigned long long idle_start = start.idle + start.iowait;
    unsigned long long idle_end   = end.idle   + end.iowait;

    unsigned long long nonidle_start = start.user + start.nice + start.system +
                                       start.irq + start.softirq + start.steal;
    unsigned long long nonidle_end   = end.user + end.nice + end.system +
                                       end.irq + end.softirq + end.steal;

    unsigned long long total_start = idle_start + nonidle_start;
    unsigned long long total_end   = idle_end   + nonidle_end;

    double totald = (double)(total_end - total_start);
    double idled  = (double)(idle_end - idle_start);

    if (totald <= 0.0) return 0.0;
    double cpu_percentage = 100.0 * (1.0 - idled / totald);
    return cpu_percentage;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Usage: ./sobel_video_profile input.mp4\n");
        return 1;
    }

    const char* input_path = argv[1];

    VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        printf("Error: Cannot open video file %s\n", input_path);
        return 1;
    }

    int width  = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double src_fps = cap.get(CAP_PROP_FPS);

    int threads = 0;
    char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        threads = atoi(env_threads);
    }

    printf("# sobel_video_profile\n");
    printf("# input,%s\n", input_path);
    printf("# width,%d\n", width);
    printf("# height,%d\n", height);
    printf("# src_fps,%.2f\n", src_fps);
    printf("# omp_threads,%d\n", threads);
    printf("# columns: frame_index,compute_time_sec\n");

    Mat frame, gray, sobel_mat(height, width, CV_8UC1);
    uint8_t *dest = sobel_mat.data;

    long long frame_count = 0;
    double total_compute_time = 0.0;

    CpuTimes cpu_start{}, cpu_end{};
    read_cpu_times(cpu_start);

    double t_start = omp_get_wtime();
    double last_time = t_start;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) break;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        double c_start = omp_get_wtime();
        sobel_filter((uint8_t*)gray.data, dest, height, width, 150, 1);
        double c_end = omp_get_wtime();

        double compute_time = c_end - c_start;
        total_compute_time += compute_time;
        frame_count++;
        last_time = c_end;

        printf("%lld,%.6f\n", frame_count, compute_time);
    }

    double t_end = last_time;
    read_cpu_times(cpu_end);

    double wall_time = t_end - t_start;
    double avg_compute = (frame_count > 0) ? (total_compute_time / frame_count) : 0.0;
    double avg_fps = (wall_time > 0.0 && frame_count > 0) ? (frame_count / wall_time) : 0.0;
    double cpu_usage = compute_cpu_usage(cpu_start, cpu_end);

    printf("# summary_frames,%lld\n", frame_count);
    printf("# summary_wall_time_sec,%.6f\n", wall_time);
    printf("# summary_avg_compute_time_sec,%.6f\n", avg_compute);
    printf("# summary_avg_fps,%.2f\n", avg_fps);
    printf("# summary_cpu_usage_percent,%.1f\n", cpu_usage);

    cap.release();
    return 0;
}
