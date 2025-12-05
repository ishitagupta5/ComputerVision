#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>   // for access()

// Check if a directory exists
bool dir_exists(const char* path) {
    struct stat st;
    return (stat(path, &st) == 0) && S_ISDIR(st.st_mode);
}

// Create directory if needed
void ensure_dir(const char* path) {
    if (!dir_exists(path)) {
        std::string cmd = "mkdir -p \"";
        cmd += path;
        cmd += "\"";
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::fprintf(stderr, "ERROR: failed to create directory %s\n", path);
            std::exit(1);
        }
    }
}

// Check if a file exists
bool file_exists(const char* path) {
    return access(path, F_OK) == 0;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s input.mp4 output.mp4\n", argv[0]);
        return 1;
    }

    std::string input_video  = argv[1];
    std::string output_video = argv[2];

    const char* frames_dir = "frames";
    const char* edges_dir  = "edges";

    ensure_dir(frames_dir);
    ensure_dir(edges_dir);

    // 1) Use ffmpeg to extract grayscale frames
    //
    // -y            : overwrite without asking
    // -loglevel error : suppress vomit, only show real errors
    // -vf format=gray : convert to grayscale
    //
    std::string cmd_extract =
        "ffmpeg -y -loglevel error "
        "-i \"" + input_video + "\" "
        "-vf format=gray "
        "\"" + std::string(frames_dir) + "/frame_%05d.png\"";

    std::printf("Extracting frames from %s ...\n", input_video.c_str());
    int ret = std::system(cmd_extract.c_str());
    if (ret != 0) {
        std::fprintf(stderr, "ERROR: ffmpeg frame extraction failed (code %d)\n", ret);
        return 1;
    }

    // 2) Run Sobel CUDA on each frame
    //
    int frame_idx = 1;
    int processed = 0;

    while (true) {
        char in_name[256];
        char out_name[256];

        std::snprintf(in_name, sizeof(in_name), "%s/frame_%05d.png", frames_dir, frame_idx);
        std::snprintf(out_name, sizeof(out_name), "%s/frame_%05d.png", edges_dir, frame_idx);

        if (!file_exists(in_name)) {
            // No more frames
            break;
        }

       // call sobel_cuda with verbose = 0 (quiet)
std::string cmd_sobel =
    "./sobel_cuda \"" + std::string(in_name) + "\" \"" + std::string(out_name) + "\" 0";

// only print for first frame and every 300th frame
if (frame_idx == 1 || frame_idx % 300 == 0) {
    std::printf("Processing frame %d: %s -> %s\n", frame_idx, in_name, out_name);
}

ret = std::system(cmd_sobel.c_str());

        if (ret != 0) {
            std::fprintf(stderr, "ERROR: sobel_cuda failed on %s (code %d)\n", in_name, ret);
            return 1;
        }

        ++frame_idx;
        ++processed;
    }

    if (processed == 0) {
        std::fprintf(stderr, "ERROR: no frames found in %s\n", frames_dir);
        return 1;
    }

    std::printf("Processed %d frames with CUDA Sobel\n", processed);

    // 3) Re-encode processed frames back into an mp4
    //
    // We’ll assume ~30 fps; adjust if you want.
    // Use mpeg4 encoder since libx264 is not available on your system.
    //
    int fps = 30;
    char fps_str[32];
    std::snprintf(fps_str, sizeof(fps_str), "%d", fps);

    std::string cmd_encode =
        "ffmpeg -y -loglevel error "
        "-framerate " + std::string(fps_str) + " "
        "-i \"" + std::string(edges_dir) + "/frame_%05d.png\" "
        "-c:v mpeg4 -pix_fmt yuv420p "
        "\"" + output_video + "\"";

    std::printf("Encoding Sobel video to %s ...\n", output_video.c_str());
    ret = std::system(cmd_encode.c_str());
    if (ret != 0) {
        std::fprintf(stderr, "ERROR: ffmpeg encode failed (code %d)\n", ret);
        return 1;
    }

    std::printf("Done. Saved GPU Sobel video to %s\n", output_video.c_str());
    return 0;
}
