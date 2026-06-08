#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define SP_MEDIAN_API __declspec(dllexport)
#else
#define SP_MEDIAN_API
#endif

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

static void convolve_float32_slab(
    const float* const* slices,
    float* output,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    const int plane = width * height;
    const int pad_x = kernel_width / 2;
    const int pad_y = kernel_height / 2;
    const int pad_z = kernel_depth / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        float* output_slice = output + out_z * plane;

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;

                for (int kz = 0; kz < kernel_depth; ++kz) {
                    const int zz = center_z + kz - pad_z;
                    if (zz < 0 || zz >= window_length) {
                        continue;
                    }

                    const float* input_slice = slices[zz];
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int yy = y + ky - pad_y;
                        if (yy < 0 || yy >= height) {
                            continue;
                        }

                        const int source_row = yy * width;
                        const int kernel_row = (kz * kernel_height + ky) * kernel_width;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            const int xx = x + kx - pad_x;
                            if (0 <= xx && xx < width) {
                                acc += input_slice[source_row + xx] * kernel[kernel_row + kx];
                            }
                        }
                    }
                }

                output_slice[y * width + x] = acc;
            }
        }
    }
}

static void convolve_float32_slice_outputs(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    const int pad_x = kernel_width / 2;
    const int pad_y = kernel_height / 2;
    const int pad_z = kernel_depth / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        float* output_slice = outputs[out_z];

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;

                for (int kz = 0; kz < kernel_depth; ++kz) {
                    const int zz = center_z + kz - pad_z;
                    if (zz < 0 || zz >= window_length) {
                        continue;
                    }

                    const float* input_slice = slices[zz];
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int yy = y + ky - pad_y;
                        if (yy < 0 || yy >= height) {
                            continue;
                        }

                        const int source_row = yy * width;
                        const int kernel_row = (kz * kernel_height + ky) * kernel_width;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            const int xx = x + kx - pad_x;
                            if (0 <= xx && xx < width) {
                                acc += input_slice[source_row + xx] * kernel[kernel_row + kx];
                            }
                        }
                    }
                }

                output_slice[y * width + x] = acc;
            }
        }
    }
}

static inline uint8_t clamp_round_uint8(float value)
{
    if (std::isnan(value) || value <= 0.0f) {
        return 0;
    }
    if (value >= 255.0f) {
        return 255;
    }
    return static_cast<uint8_t>(std::nearbyint(value));
}

static inline uint8_t convolve_uint8_guarded_pixel(
    const uint8_t* const* slices,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int pad_x,
    int pad_y,
    int pad_z,
    int center_z,
    int x,
    int y)
{
    float acc = 0.0f;

    for (int kz = 0; kz < kernel_depth; ++kz) {
        const int zz = center_z + kz - pad_z;
        if (zz < 0 || zz >= window_length) {
            continue;
        }

        const uint8_t* input_slice = slices[zz];
        for (int ky = 0; ky < kernel_height; ++ky) {
            const int yy = y + ky - pad_y;
            if (yy < 0 || yy >= height) {
                continue;
            }

            const int source_row = yy * width;
            const int kernel_row = (kz * kernel_height + ky) * kernel_width;
            for (int kx = 0; kx < kernel_width; ++kx) {
                const int xx = x + kx - pad_x;
                if (0 <= xx && xx < width) {
                    acc += static_cast<float>(input_slice[source_row + xx]) * kernel[kernel_row + kx];
                }
            }
        }
    }

    return clamp_round_uint8(acc);
}

static void convolve_uint8_slice_outputs(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    const int pad_x = kernel_width / 2;
    const int pad_y = kernel_height / 2;
    const int pad_z = kernel_depth / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        uint8_t* output_slice = outputs[out_z];
        const bool full_z = 0 <= center_z - pad_z && center_z + pad_z < window_length;
        const int x_begin = pad_x;
        const int x_end = width - pad_x;
        const int y_begin = pad_y;
        const int y_end = height - pad_y;
        const bool has_interior = full_z && x_begin < x_end && y_begin < y_end;

        if (!has_interior) {
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    output_slice[y * width + x] =
                        convolve_uint8_guarded_pixel(
                            slices,
                            kernel,
                            width,
                            height,
                            window_length,
                            kernel_width,
                            kernel_height,
                            kernel_depth,
                            pad_x,
                            pad_y,
                            pad_z,
                            center_z,
                            x,
                            y);
                }
            }
            continue;
        }

        for (int y = 0; y < y_begin; ++y) {
            for (int x = 0; x < width; ++x) {
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        height,
                        window_length,
                        kernel_width,
                        kernel_height,
                        kernel_depth,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y);
            }
        }

        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < x_begin; ++x) {
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        height,
                        window_length,
                        kernel_width,
                        kernel_height,
                        kernel_depth,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y);
            }

            for (int x = x_begin; x < x_end; ++x) {
                float acc = 0.0f;

                for (int kz = 0; kz < kernel_depth; ++kz) {
                    const uint8_t* input_slice = slices[center_z + kz - pad_z];
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int source_row = (y + ky - pad_y) * width;
                        const int kernel_row = (kz * kernel_height + ky) * kernel_width;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            acc += static_cast<float>(input_slice[source_row + x + kx - pad_x]) *
                                   kernel[kernel_row + kx];
                        }
                    }
                }

                output_slice[y * width + x] = clamp_round_uint8(acc);
            }

            for (int x = x_end; x < width; ++x) {
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        height,
                        window_length,
                        kernel_width,
                        kernel_height,
                        kernel_depth,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y);
            }
        }

        for (int y = y_end; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        height,
                        window_length,
                        kernel_width,
                        kernel_height,
                        kernel_depth,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y);
            }
        }
    }
}

extern "C" {

SP_MEDIAN_API void sp_median_uint8_nth_slab(
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

SP_MEDIAN_API void sp_median_uint16_nth_slab(
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

SP_MEDIAN_API void sp_median_int32_nth_slab(
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

SP_MEDIAN_API void sp_median_float32_nth_slab(
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

SP_MEDIAN_API void sp_convolve_float32_slab(
    const float* const* slices,
    float* output,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    convolve_float32_slab(
        slices,
        output,
        kernel,
        width,
        height,
        window_length,
        kernel_width,
        kernel_height,
        kernel_depth,
        output_start,
        output_count);
}

SP_MEDIAN_API void sp_convolve_float32_slices(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    convolve_float32_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        window_length,
        kernel_width,
        kernel_height,
        kernel_depth,
        output_start,
        output_count);
}

SP_MEDIAN_API void sp_convolve_uint8_slices(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_width,
    int kernel_height,
    int kernel_depth,
    int output_start,
    int output_count)
{
    convolve_uint8_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        window_length,
        kernel_width,
        kernel_height,
        kernel_depth,
        output_start,
        output_count);
}

}
