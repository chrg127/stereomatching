#include "image.h"

#include "ghost.h"

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
#ifndef NO_WRITES
    size_t elem_size = imtype_to_size(type);
    size_t array_size = (width + ghost_size * 2) * (height + ghost_size * 2) * elem_size;
    void *host_data = make_host_copy(ghost_to_real(data, elem_size, width, ghost_size), array_size);
    write_image(real_to_ghost(host_data, elem_size, width, ghost_size), width, height, ghost_size, type, name, number);
#endif
}
