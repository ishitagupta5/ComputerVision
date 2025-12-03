#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <stdlib.h>

void resize_image(const char* in, const char* out, int new_w, int new_h) {
    int w, h, c;
    unsigned char* img = stbi_load(in, &w, &h, &c, 3);
    if (!img) return;

    unsigned char* resized = (unsigned char*)malloc(new_w * new_h * 3);

    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            int src_x = x * w / new_w;
            int src_y = y * h / new_h;
            for (int k = 0; k < 3; k++) {
                resized[(y * new_w + x) * 3 + k] =
                    img[(src_y * w + src_x) * 3 + k];
            }
        }
    }

    stbi_write_png(out, new_w, new_h, 3, resized, new_w * 3);

    free(img);
    free(resized);
}

int main() {
    resize_image("fox.png", "fox_256.png", 256, 256);
    resize_image("fox.png", "fox_512.png", 512, 512);
    resize_image("fox.png", "fox_1024.png", 1024, 1024);
    resize_image("fox.png", "fox_2048.png", 2048, 2048);
    return 0;
}
