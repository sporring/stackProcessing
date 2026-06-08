#include <algorithm>
#include <cstdint>
#include <vector>

template <typename T>
static void median_nth_slab(
    const T* const* slices,
    T* output,
    int width,
    int height,
    int window_length,
    int radius,
    int output_start,
    int output_count)
{
    const int diameter = 2 * radius + 1;
    const int count = diameter * diameter * diameter;
    const int median = count / 2;
    const int plane = width * height;
    std::vector<T> scratch;
    scratch.resize(static_cast<size_t>(count));

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        T* output_slice = output + out_z * plane;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int k = 0;
                for (int dz = -radius; dz <= radius; ++dz) {
                    const int zz = center_z + dz;
                    const T* input_slice =
                        (0 <= zz && zz < window_length) ? slices[zz] : nullptr;

                    for (int dy = -radius; dy <= radius; ++dy) {
                        const int yy = y + dy;
                        for (int dx = -radius; dx <= radius; ++dx) {
                            const int xx = x + dx;
                            if (input_slice != nullptr &&
                                0 <= xx && xx < width &&
                                0 <= yy && yy < height) {
                                scratch[static_cast<size_t>(k)] = input_slice[yy * width + xx];
                            } else {
                                scratch[static_cast<size_t>(k)] = static_cast<T>(0);
                            }
                            ++k;
                        }
                    }
                }

                auto first = scratch.begin();
                auto nth = first + median;
                auto last = first + count;
                std::nth_element(first, nth, last);
                output_slice[y * width + x] = *nth;
            }
        }
    }
}

extern "C" {

void sp_median_uint8_nth_slab(
    const uint8_t* const* slices,
    uint8_t* output,
    int width,
    int height,
    int window_length,
    int radius,
    int output_start,
    int output_count)
{
    median_nth_slab<uint8_t>(
        slices,
        output,
        width,
        height,
        window_length,
        radius,
        output_start,
        output_count);
}

void sp_median_uint16_nth_slab(
    const uint16_t* const* slices,
    uint16_t* output,
    int width,
    int height,
    int window_length,
    int radius,
    int output_start,
    int output_count)
{
    median_nth_slab<uint16_t>(
        slices,
        output,
        width,
        height,
        window_length,
        radius,
        output_start,
        output_count);
}

void sp_median_int32_nth_slab(
    const int32_t* const* slices,
    int32_t* output,
    int width,
    int height,
    int window_length,
    int radius,
    int output_start,
    int output_count)
{
    median_nth_slab<int32_t>(
        slices,
        output,
        width,
        height,
        window_length,
        radius,
        output_start,
        output_count);
}

void sp_median_float32_nth_slab(
    const float* const* slices,
    float* output,
    int width,
    int height,
    int window_length,
    int radius,
    int output_start,
    int output_count)
{
    median_nth_slab<float>(
        slices,
        output,
        width,
        height,
        window_length,
        radius,
        output_start,
        output_count);
}

}
