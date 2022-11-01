#include "image.h"

int imtype_to_size(ImageType type)
{
    switch (type) {
    case IMTYPE_BINARY:     return sizeof(u8);
    case IMTYPE_GRAY_FLOAT: return sizeof(double);
    case IMTYPE_GRAY_INT:   return sizeof(i32);
    default:                return 0;
    }
}

void write_image_from_gpu(void *data, int width, int height, int ghost_size, ImageType type, const char *name, int number)
{
    void *real_data = xmalloc(width * height * imtype_to_size(type));
    cudaMemcpy(real_data, data, width * height * imtype_to_size(type), cudaMemcpyDeviceToHost);
    write_image(real_data, width, height, ghost_size, type, name, number);
}
