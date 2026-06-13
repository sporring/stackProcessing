#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>
#include <vector>
#include <fftw3.h>

#if defined(_WIN32)
#define SP_LOWLEVEL_API __declspec(dllexport)
#else
#define SP_LOWLEVEL_API
#endif

static std::mutex fftwf_planner_mutex;

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
                        if (input_slice == nullptr || yy < 0 || yy >= height) {
                            for (int dx = -radius; dx <= radius; ++dx) {
                                scratch[static_cast<size_t>(k)] = static_cast<T>(0);
                                ++k;
                            }
                        } else {
                            const int row = yy * width;
                            for (int dx = -radius; dx <= radius; ++dx) {
                                const int xx = x + dx;
                                scratch[static_cast<size_t>(k)] =
                                    (0 <= xx && xx < width) ? input_slice[row + xx] : static_cast<T>(0);
                                ++k;
                            }
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
    int kernel_width,
    int kernel_height,
    int pad_x,
    int pad_y,
    int pad_z,
    int center_z,
    int x,
    int y,
    int kz_begin,
    int kz_end,
    int ky_begin,
    int ky_end,
    int kx_begin,
    int kx_end)
{
    float acc = 0.0f;

    for (int kz = kz_begin; kz < kz_end; ++kz) {
        const int zz = center_z + kz - pad_z;
        const uint8_t* input_slice = slices[zz];
        for (int ky = ky_begin; ky < ky_end; ++ky) {
            const int yy = y + ky - pad_y;
            const int source_row = yy * width;
            const int kernel_row = (kz * kernel_height + ky) * kernel_width;
            for (int kx = kx_begin; kx < kx_end; ++kx) {
                const int xx = x + kx - pad_x;
                acc += static_cast<float>(input_slice[source_row + xx]) * kernel[kernel_row + kx];
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
    const int x_begin = pad_x;
    const int x_end = width - pad_x;
    const int y_begin = pad_y;
    const int y_end = height - pad_y;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        uint8_t* output_slice = outputs[out_z];
        const bool full_z = 0 <= center_z - pad_z && center_z + pad_z < window_length;
        const bool has_interior = full_z && x_begin < x_end && y_begin < y_end;
        const int kz_begin = full_z ? 0 : std::max(0, pad_z - center_z);
        const int kz_end = full_z ? kernel_depth : std::min(kernel_depth, window_length - center_z + pad_z);

        if (!has_interior) {
            for (int y = 0; y < height; ++y) {
                const int ky_begin = std::max(0, pad_y - y);
                const int ky_end = std::min(kernel_height, height - y + pad_y);
                for (int x = 0; x < width; ++x) {
                    const int kx_begin = std::max(0, pad_x - x);
                    const int kx_end = std::min(kernel_width, width - x + pad_x);
                    output_slice[y * width + x] =
                        convolve_uint8_guarded_pixel(
                            slices,
                            kernel,
                            width,
                            kernel_width,
                            kernel_height,
                            pad_x,
                            pad_y,
                            pad_z,
                            center_z,
                            x,
                            y,
                            kz_begin,
                            kz_end,
                            ky_begin,
                            ky_end,
                            kx_begin,
                            kx_end);
                }
            }
            continue;
        }

        for (int y = 0; y < y_begin; ++y) {
            const int ky_begin = std::max(0, pad_y - y);
            const int ky_end = std::min(kernel_height, height - y + pad_y);
            for (int x = 0; x < width; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        ky_begin,
                        ky_end,
                        kx_begin,
                        kx_end);
            }
        }

        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < x_begin; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        0,
                        kernel_height,
                        kx_begin,
                        kx_end);
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
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        0,
                        kernel_height,
                        kx_begin,
                        kx_end);
            }
        }

        for (int y = y_end; y < height; ++y) {
            const int ky_begin = std::max(0, pad_y - y);
            const int ky_end = std::min(kernel_height, height - y + pad_y);
            for (int x = 0; x < width; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_uint8_guarded_pixel(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        ky_begin,
                        ky_end,
                        kx_begin,
                        kx_end);
            }
        }
    }
}

static void convolve_uint8_x_slice_outputs(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        uint8_t* output_slice = outputs[out_z];

        if (center_z < 0 || center_z >= window_length) {
            std::fill(output_slice, output_slice + width * height, 0);
            continue;
        }

        const uint8_t* input_slice = slices[center_z];
        const int x_begin = pad;
        const int x_end = width - pad;

        for (int y = 0; y < height; ++y) {
            const int row = y * width;

            for (int x = 0; x < x_begin && x < width; ++x) {
                float acc = 0.0f;
                const int k_begin = std::max(0, pad - x);
                const int k_end = std::min(kernel_length, width - x + pad);
                for (int k = k_begin; k < k_end; ++k) {
                    const int xx = x + k - pad;
                    acc += static_cast<float>(input_slice[row + xx]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }

            for (int x = x_begin; x < x_end; ++x) {
                float acc = 0.0f;
                const int source = row + x - pad;
                for (int k = 0; k < kernel_length; ++k) {
                    acc += static_cast<float>(input_slice[source + k]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }

            for (int x = std::max(x_begin, x_end); x < width; ++x) {
                float acc = 0.0f;
                const int k_begin = std::max(0, pad - x);
                const int k_end = std::min(kernel_length, width - x + pad);
                for (int k = k_begin; k < k_end; ++k) {
                    const int xx = x + k - pad;
                    acc += static_cast<float>(input_slice[row + xx]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }
        }
    }
}

static void convolve_uint8_y_slice_outputs(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        uint8_t* output_slice = outputs[out_z];

        if (center_z < 0 || center_z >= window_length) {
            std::fill(output_slice, output_slice + width * height, 0);
            continue;
        }

        const uint8_t* input_slice = slices[center_z];
        const int y_begin = pad;
        const int y_end = height - pad;

        for (int y = 0; y < y_begin && y < height; ++y) {
            const int row = y * width;
            const int k_begin = std::max(0, pad - y);
            const int k_end = std::min(kernel_length, height - y + pad);
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = k_begin; k < k_end; ++k) {
                    const int yy = y + k - pad;
                    acc += static_cast<float>(input_slice[yy * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }
        }

        for (int y = y_begin; y < y_end; ++y) {
            const int row = y * width;
            const int source = (y - pad) * width;
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = 0; k < kernel_length; ++k) {
                    acc += static_cast<float>(input_slice[source + k * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }
        }

        for (int y = std::max(y_begin, y_end); y < height; ++y) {
            const int row = y * width;
            const int k_begin = std::max(0, pad - y);
            const int k_end = std::min(kernel_length, height - y + pad);
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = k_begin; k < k_end; ++k) {
                    const int yy = y + k - pad;
                    acc += static_cast<float>(input_slice[yy * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_uint8(acc);
            }
        }
    }
}

static void convolve_uint8_z_slice_outputs(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;
    const int plane = width * height;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        uint8_t* output_slice = outputs[out_z];
        const int k_begin = std::max(0, pad - center_z);
        const int k_end = std::min(kernel_length, window_length - center_z + pad);

        for (int i = 0; i < plane; ++i) {
            float acc = 0.0f;
            for (int k = k_begin; k < k_end; ++k) {
                const int zz = center_z + k - pad;
                acc += static_cast<float>(slices[zz][i]) * kernel[k];
            }
            output_slice[i] = clamp_round_uint8(acc);
        }
    }
}

template <typename T>
static inline T clamp_round_scalar(float value)
{
    return static_cast<T>(std::nearbyint(value));
}

template <>
inline int8_t clamp_round_scalar<int8_t>(float value)
{
    if (std::isnan(value)) {
        return 0;
    }
    if (value <= -128.0f) {
        return static_cast<int8_t>(-128);
    }
    if (value >= 127.0f) {
        return static_cast<int8_t>(127);
    }
    return static_cast<int8_t>(std::nearbyint(value));
}

template <>
inline uint8_t clamp_round_scalar<uint8_t>(float value)
{
    return clamp_round_uint8(value);
}

template <>
inline uint16_t clamp_round_scalar<uint16_t>(float value)
{
    if (std::isnan(value) || value <= 0.0f) {
        return 0;
    }
    if (value >= 65535.0f) {
        return 65535;
    }
    return static_cast<uint16_t>(std::nearbyint(value));
}

template <>
inline int16_t clamp_round_scalar<int16_t>(float value)
{
    if (std::isnan(value)) {
        return 0;
    }
    if (value <= static_cast<float>(INT16_MIN)) {
        return INT16_MIN;
    }
    if (value >= static_cast<float>(INT16_MAX)) {
        return INT16_MAX;
    }
    return static_cast<int16_t>(std::nearbyint(value));
}

template <>
inline int32_t clamp_round_scalar<int32_t>(float value)
{
    if (std::isnan(value)) {
        return 0;
    }
    if (value <= static_cast<float>(INT32_MIN)) {
        return INT32_MIN;
    }
    if (value >= static_cast<float>(INT32_MAX)) {
        return INT32_MAX;
    }
    return static_cast<int32_t>(std::nearbyint(value));
}

template <>
inline float clamp_round_scalar<float>(float value)
{
    return value;
}

template <typename T>
static inline T convolve_scalar_guarded_pixel(
    const T* const* slices,
    const float* kernel,
    int width,
    int kernel_width,
    int kernel_height,
    int pad_x,
    int pad_y,
    int pad_z,
    int center_z,
    int x,
    int y,
    int kz_begin,
    int kz_end,
    int ky_begin,
    int ky_end,
    int kx_begin,
    int kx_end)
{
    float acc = 0.0f;

    for (int kz = kz_begin; kz < kz_end; ++kz) {
        const int zz = center_z + kz - pad_z;
        const T* input_slice = slices[zz];
        for (int ky = ky_begin; ky < ky_end; ++ky) {
            const int yy = y + ky - pad_y;
            const int source_row = yy * width;
            const int kernel_row = (kz * kernel_height + ky) * kernel_width;
            for (int kx = kx_begin; kx < kx_end; ++kx) {
                const int xx = x + kx - pad_x;
                acc += static_cast<float>(input_slice[source_row + xx]) * kernel[kernel_row + kx];
            }
        }
    }

    return clamp_round_scalar<T>(acc);
}

template <typename T>
static void convolve_scalar_slice_outputs(
    const T* const* slices,
    T* const* outputs,
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
    const int x_begin = pad_x;
    const int x_end = width - pad_x;
    const int y_begin = pad_y;
    const int y_end = height - pad_y;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        T* output_slice = outputs[out_z];
        const bool full_z = 0 <= center_z - pad_z && center_z + pad_z < window_length;
        const bool has_interior = full_z && x_begin < x_end && y_begin < y_end;
        const int kz_begin = full_z ? 0 : std::max(0, pad_z - center_z);
        const int kz_end = full_z ? kernel_depth : std::min(kernel_depth, window_length - center_z + pad_z);

        if (!has_interior) {
            for (int y = 0; y < height; ++y) {
                const int ky_begin = std::max(0, pad_y - y);
                const int ky_end = std::min(kernel_height, height - y + pad_y);
                for (int x = 0; x < width; ++x) {
                    const int kx_begin = std::max(0, pad_x - x);
                    const int kx_end = std::min(kernel_width, width - x + pad_x);
                    output_slice[y * width + x] =
                        convolve_scalar_guarded_pixel<T>(
                            slices,
                            kernel,
                            width,
                            kernel_width,
                            kernel_height,
                            pad_x,
                            pad_y,
                            pad_z,
                            center_z,
                            x,
                            y,
                            kz_begin,
                            kz_end,
                            ky_begin,
                            ky_end,
                            kx_begin,
                            kx_end);
                }
            }
            continue;
        }

        for (int y = 0; y < y_begin; ++y) {
            const int ky_begin = std::max(0, pad_y - y);
            const int ky_end = std::min(kernel_height, height - y + pad_y);
            for (int x = 0; x < width; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_scalar_guarded_pixel<T>(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        ky_begin,
                        ky_end,
                        kx_begin,
                        kx_end);
            }
        }

        for (int y = y_begin; y < y_end; ++y) {
            for (int x = 0; x < x_begin; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_scalar_guarded_pixel<T>(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        0,
                        kernel_height,
                        kx_begin,
                        kx_end);
            }

            for (int x = x_begin; x < x_end; ++x) {
                float acc = 0.0f;

                for (int kz = 0; kz < kernel_depth; ++kz) {
                    const T* input_slice = slices[center_z + kz - pad_z];
                    for (int ky = 0; ky < kernel_height; ++ky) {
                        const int source_row = (y + ky - pad_y) * width;
                        const int kernel_row = (kz * kernel_height + ky) * kernel_width;
                        for (int kx = 0; kx < kernel_width; ++kx) {
                            acc += static_cast<float>(input_slice[source_row + x + kx - pad_x]) *
                                   kernel[kernel_row + kx];
                        }
                    }
                }

                output_slice[y * width + x] = clamp_round_scalar<T>(acc);
            }

            for (int x = x_end; x < width; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_scalar_guarded_pixel<T>(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        0,
                        kernel_height,
                        kx_begin,
                        kx_end);
            }
        }

        for (int y = y_end; y < height; ++y) {
            const int ky_begin = std::max(0, pad_y - y);
            const int ky_end = std::min(kernel_height, height - y + pad_y);
            for (int x = 0; x < width; ++x) {
                const int kx_begin = std::max(0, pad_x - x);
                const int kx_end = std::min(kernel_width, width - x + pad_x);
                output_slice[y * width + x] =
                    convolve_scalar_guarded_pixel<T>(
                        slices,
                        kernel,
                        width,
                        kernel_width,
                        kernel_height,
                        pad_x,
                        pad_y,
                        pad_z,
                        center_z,
                        x,
                        y,
                        kz_begin,
                        kz_end,
                        ky_begin,
                        ky_end,
                        kx_begin,
                        kx_end);
            }
        }
    }
}

template <typename T>
static void convolve_axis_x_slice_outputs(
    const T* const* slices,
    T* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        T* output_slice = outputs[out_z];

        if (center_z < 0 || center_z >= window_length) {
            std::fill(output_slice, output_slice + width * height, static_cast<T>(0));
            continue;
        }

        const T* input_slice = slices[center_z];
        const int x_begin = pad;
        const int x_end = width - pad;

        for (int y = 0; y < height; ++y) {
            const int row = y * width;

            for (int x = 0; x < x_begin && x < width; ++x) {
                float acc = 0.0f;
                const int k_begin = std::max(0, pad - x);
                const int k_end = std::min(kernel_length, width - x + pad);
                for (int k = k_begin; k < k_end; ++k) {
                    const int xx = x + k - pad;
                    acc += static_cast<float>(input_slice[row + xx]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }

            for (int x = x_begin; x < x_end; ++x) {
                float acc = 0.0f;
                const int source = row + x - pad;
                for (int k = 0; k < kernel_length; ++k) {
                    acc += static_cast<float>(input_slice[source + k]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }

            for (int x = std::max(x_begin, x_end); x < width; ++x) {
                float acc = 0.0f;
                const int k_begin = std::max(0, pad - x);
                const int k_end = std::min(kernel_length, width - x + pad);
                for (int k = k_begin; k < k_end; ++k) {
                    const int xx = x + k - pad;
                    acc += static_cast<float>(input_slice[row + xx]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }
        }
    }
}

template <typename T>
static void convolve_axis_y_slice_outputs(
    const T* const* slices,
    T* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        T* output_slice = outputs[out_z];

        if (center_z < 0 || center_z >= window_length) {
            std::fill(output_slice, output_slice + width * height, static_cast<T>(0));
            continue;
        }

        const T* input_slice = slices[center_z];
        const int y_begin = pad;
        const int y_end = height - pad;

        for (int y = 0; y < y_begin && y < height; ++y) {
            const int row = y * width;
            const int k_begin = std::max(0, pad - y);
            const int k_end = std::min(kernel_length, height - y + pad);
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = k_begin; k < k_end; ++k) {
                    const int yy = y + k - pad;
                    acc += static_cast<float>(input_slice[yy * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }
        }

        for (int y = y_begin; y < y_end; ++y) {
            const int row = y * width;
            const int source = (y - pad) * width;
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = 0; k < kernel_length; ++k) {
                    acc += static_cast<float>(input_slice[source + k * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }
        }

        for (int y = std::max(y_begin, y_end); y < height; ++y) {
            const int row = y * width;
            const int k_begin = std::max(0, pad - y);
            const int k_end = std::min(kernel_length, height - y + pad);
            for (int x = 0; x < width; ++x) {
                float acc = 0.0f;
                for (int k = k_begin; k < k_end; ++k) {
                    const int yy = y + k - pad;
                    acc += static_cast<float>(input_slice[yy * width + x]) * kernel[k];
                }
                output_slice[row + x] = clamp_round_scalar<T>(acc);
            }
        }
    }
}

template <typename T>
static void convolve_axis_z_slice_outputs(
    const T* const* slices,
    T* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    const int pad = kernel_length / 2;
    const int plane = width * height;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        T* output_slice = outputs[out_z];
        const int k_begin = std::max(0, pad - center_z);
        const int k_end = std::min(kernel_length, window_length - center_z + pad);

        for (int i = 0; i < plane; ++i) {
            float acc = 0.0f;
            for (int k = k_begin; k < k_end; ++k) {
                const int zz = center_z + k - pad;
                acc += static_cast<float>(slices[zz][i]) * kernel[k];
            }
            output_slice[i] = clamp_round_scalar<T>(acc);
        }
    }
}

static int convolve_float32_vector_components_slice_outputs(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int components,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count,
    int axis)
{
    if (slices == nullptr || outputs == nullptr || kernel == nullptr ||
        width <= 0 || height <= 0 || components <= 0 ||
        window_length <= 0 || kernel_length <= 0 || output_count < 0) {
        return 1;
    }
    if ((kernel_length % 2) == 0) {
        return 2;
    }
    if (axis < 0 || axis > 2) {
        return 3;
    }

    const int pad = kernel_length / 2;
    const int vector_plane = width * height * components;

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int center_z = output_start + out_z;
        float* output_slice = outputs[out_z];
        if (output_slice == nullptr) {
            return 4;
        }

        if (axis == 0 || axis == 1) {
            if (center_z < 0 || center_z >= window_length || slices[center_z] == nullptr) {
                std::fill(output_slice, output_slice + vector_plane, 0.0f);
                continue;
            }

            const float* input_slice = slices[center_z];
            if (axis == 0) {
                const int x_begin = pad;
                const int x_end = width - pad;

                for (int y = 0; y < height; ++y) {
                    const int row = y * width;

                    for (int x = 0; x < x_begin && x < width; ++x) {
                        const int target = (row + x) * components;
                        const int k_begin = std::max(0, pad - x);
                        const int k_end = std::min(kernel_length, width - x + pad);
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = k_begin; k < k_end; ++k) {
                                const int xx = x + k - pad;
                                acc += input_slice[(row + xx) * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }

                    for (int x = x_begin; x < x_end; ++x) {
                        const int target = (row + x) * components;
                        const int source = (row + x - pad) * components;
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = 0; k < kernel_length; ++k) {
                                acc += input_slice[source + k * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }

                    for (int x = std::max(x_begin, x_end); x < width; ++x) {
                        const int target = (row + x) * components;
                        const int k_begin = std::max(0, pad - x);
                        const int k_end = std::min(kernel_length, width - x + pad);
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = k_begin; k < k_end; ++k) {
                                const int xx = x + k - pad;
                                acc += input_slice[(row + xx) * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }
                }
            } else {
                const int y_begin = pad;
                const int y_end = height - pad;

                for (int y = 0; y < y_begin && y < height; ++y) {
                    const int row = y * width;
                    const int k_begin = std::max(0, pad - y);
                    const int k_end = std::min(kernel_length, height - y + pad);
                    for (int x = 0; x < width; ++x) {
                        const int target = (row + x) * components;
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = k_begin; k < k_end; ++k) {
                                const int yy = y + k - pad;
                                acc += input_slice[(yy * width + x) * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }
                }

                for (int y = y_begin; y < y_end; ++y) {
                    const int row = y * width;
                    const int source_row = (y - pad) * width;
                    for (int x = 0; x < width; ++x) {
                        const int target = (row + x) * components;
                        const int source = (source_row + x) * components;
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = 0; k < kernel_length; ++k) {
                                acc += input_slice[source + k * width * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }
                }

                for (int y = std::max(y_begin, y_end); y < height; ++y) {
                    const int row = y * width;
                    const int k_begin = std::max(0, pad - y);
                    const int k_end = std::min(kernel_length, height - y + pad);
                    for (int x = 0; x < width; ++x) {
                        const int target = (row + x) * components;
                        for (int c = 0; c < components; ++c) {
                            float acc = 0.0f;
                            for (int k = k_begin; k < k_end; ++k) {
                                const int yy = y + k - pad;
                                acc += input_slice[(yy * width + x) * components + c] * kernel[k];
                            }
                            output_slice[target + c] = acc;
                        }
                    }
                }
            }
        } else {
            const int k_begin = std::max(0, pad - center_z);
            const int k_end = std::min(kernel_length, window_length - center_z + pad);
            for (int i = 0; i < vector_plane; ++i) {
                float acc = 0.0f;
                for (int k = k_begin; k < k_end; ++k) {
                    const int zz = center_z + k - pad;
                    const float* input_slice = slices[zz];
                    if (input_slice != nullptr) {
                        acc += input_slice[i] * kernel[k];
                    }
                }
                output_slice[i] = acc;
            }
        }
    }

    return 0;
}

static void scale_complex_buffer(float* interleaved, int complex_count, float scale)
{
    const int value_count = 2 * complex_count;
    for (int i = 0; i < value_count; ++i) {
        interleaved[i] *= scale;
    }
}

static int fftwf_complex_xy_inplace(float* interleaved, int width, int height, int inverse)
{
    if (interleaved == nullptr || width <= 0 || height <= 0) {
        return 1;
    }

    fftwf_complex* data = reinterpret_cast<fftwf_complex*>(interleaved);
    const int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
    fftwf_plan plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(fftwf_planner_mutex);
        plan = fftwf_plan_dft_2d(height, width, data, data, sign, FFTW_ESTIMATE);
    }

    if (plan == nullptr) {
        return 2;
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    if (inverse) {
        scale_complex_buffer(interleaved, width * height, 1.0f / static_cast<float>(width * height));
    }

    return 0;
}

static int fftwf_complex_z_inplace(float* interleaved, int width, int height, int depth, int inverse)
{
    if (interleaved == nullptr || width <= 0 || height <= 0 || depth <= 0) {
        return 1;
    }

    const int plane = width * height;
    fftwf_complex* data = reinterpret_cast<fftwf_complex*>(interleaved);
    int n[] = { depth };
    const int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;

    fftwf_plan plan = nullptr;
    {
        std::lock_guard<std::mutex> lock(fftwf_planner_mutex);
        plan =
            fftwf_plan_many_dft(
                1,
                n,
                plane,
                data,
                nullptr,
                plane,
                1,
                data,
                nullptr,
                plane,
                1,
                sign,
                FFTW_ESTIMATE);
    }

    if (plan == nullptr) {
        return 2;
    }

    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    if (inverse) {
        scale_complex_buffer(interleaved, plane * depth, 1.0f / static_cast<float>(depth));
    }

    return 0;
}

static constexpr float SP_DISTANCE_INF = 1.0e20f;

static inline size_t volume_index(int width, int height, int x, int y, int z)
{
    return static_cast<size_t>((z * height + y) * width + x);
}

static void distance_transform_1d_inplace(float* values, int length)
{
    if (length <= 0) {
        return;
    }

    std::vector<int> sites(static_cast<size_t>(length));
    std::vector<double> boundaries(static_cast<size_t>(length) + 1);
    std::vector<float> result(static_cast<size_t>(length));

    int k = 0;
    sites[0] = 0;
    boundaries[0] = -std::numeric_limits<double>::infinity();
    boundaries[1] = std::numeric_limits<double>::infinity();

    for (int q = 1; q < length; ++q) {
        double s = 0.0;

        while (true) {
            const int vk = sites[static_cast<size_t>(k)];
            const double fq = static_cast<double>(values[q]);
            const double fvk = static_cast<double>(values[vk]);

            if (fq >= SP_DISTANCE_INF * 0.5 && fvk >= SP_DISTANCE_INF * 0.5) {
                s = std::numeric_limits<double>::infinity();
            } else if (fvk >= SP_DISTANCE_INF * 0.5) {
                s = -std::numeric_limits<double>::infinity();
            } else if (fq >= SP_DISTANCE_INF * 0.5) {
                s = std::numeric_limits<double>::infinity();
            } else {
                s =
                    ((fq + static_cast<double>(q) * static_cast<double>(q)) -
                     (fvk + static_cast<double>(vk) * static_cast<double>(vk))) /
                    (2.0 * static_cast<double>(q - vk));
            }

            if (s > boundaries[static_cast<size_t>(k)] || k == 0) {
                break;
            }
            --k;
        }

        if (s <= boundaries[static_cast<size_t>(k)]) {
            sites[0] = q;
            boundaries[0] = -std::numeric_limits<double>::infinity();
            boundaries[1] = std::numeric_limits<double>::infinity();
            k = 0;
        } else {
            ++k;
            sites[static_cast<size_t>(k)] = q;
            boundaries[static_cast<size_t>(k)] = s;
            boundaries[static_cast<size_t>(k + 1)] = std::numeric_limits<double>::infinity();
        }
    }

    k = 0;
    for (int q = 0; q < length; ++q) {
        while (boundaries[static_cast<size_t>(k + 1)] < static_cast<double>(q)) {
            ++k;
        }

        const int site = sites[static_cast<size_t>(k)];
        const float base = values[site];
        if (base >= SP_DISTANCE_INF * 0.5f) {
            result[static_cast<size_t>(q)] = SP_DISTANCE_INF;
        } else {
            const int d = q - site;
            result[static_cast<size_t>(q)] = base + static_cast<float>(d * d);
        }
    }

    for (int i = 0; i < length; ++i) {
        values[i] = result[static_cast<size_t>(i)];
    }
}

static void exact_squared_distance_transform(
    std::vector<float>& distances,
    int width,
    int height,
    int depth)
{
    const int max_length = std::max(width, std::max(height, depth));
    std::vector<float> line(static_cast<size_t>(max_length));

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                line[static_cast<size_t>(x)] = distances[volume_index(width, height, x, y, z)];
            }
            distance_transform_1d_inplace(line.data(), width);
            for (int x = 0; x < width; ++x) {
                distances[volume_index(width, height, x, y, z)] = line[static_cast<size_t>(x)];
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                line[static_cast<size_t>(y)] = distances[volume_index(width, height, x, y, z)];
            }
            distance_transform_1d_inplace(line.data(), height);
            for (int y = 0; y < height; ++y) {
                distances[volume_index(width, height, x, y, z)] = line[static_cast<size_t>(y)];
            }
        }
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int z = 0; z < depth; ++z) {
                line[static_cast<size_t>(z)] = distances[volume_index(width, height, x, y, z)];
            }
            distance_transform_1d_inplace(line.data(), depth);
            for (int z = 0; z < depth; ++z) {
                distances[volume_index(width, height, x, y, z)] = line[static_cast<size_t>(z)];
            }
        }
    }
}

static int signed_distance_band_uint8_slices(
    const uint8_t* const* slices,
    float* const* outputs,
    int width,
    int height,
    int window_length,
    int output_start,
    int output_count,
    float band_radius)
{
    if (slices == nullptr || outputs == nullptr || width <= 0 || height <= 0 ||
        window_length <= 0 || output_start < 0 || output_count < 0 ||
        output_start + output_count > window_length || band_radius <= 0.0f) {
        return 1;
    }

    const int plane = width * height;
    const size_t voxel_count = static_cast<size_t>(plane) * static_cast<size_t>(window_length);
    std::vector<float> to_foreground(voxel_count);
    std::vector<float> to_background(voxel_count);

    for (int z = 0; z < window_length; ++z) {
        const uint8_t* slice = slices[z];
        if (slice == nullptr) {
            return 2;
        }

        for (int i = 0; i < plane; ++i) {
            const bool foreground = slice[i] != 0;
            const size_t index = static_cast<size_t>(z * plane + i);
            to_foreground[index] = foreground ? 0.0f : SP_DISTANCE_INF;
            to_background[index] = foreground ? SP_DISTANCE_INF : 0.0f;
        }
    }

    exact_squared_distance_transform(to_foreground, width, height, window_length);
    exact_squared_distance_transform(to_background, width, height, window_length);

    const float nan = std::numeric_limits<float>::quiet_NaN();

    for (int out_z = 0; out_z < output_count; ++out_z) {
        const int z = output_start + out_z;
        float* output = outputs[out_z];
        if (output == nullptr) {
            return 3;
        }

        const uint8_t* source = slices[z];
        for (int i = 0; i < plane; ++i) {
            const size_t index = static_cast<size_t>(z * plane + i);
            const bool foreground = source[i] != 0;
            const float squared = foreground ? to_background[index] : to_foreground[index];

            if (squared >= SP_DISTANCE_INF * 0.5f) {
                output[i] = nan;
            } else {
                const float distance = std::sqrt(std::max(0.0f, squared));
                const float signed_distance = foreground ? -distance : distance;
                output[i] = (std::fabs(signed_distance) < band_radius) ? signed_distance : nan;
            }
        }
    }

    return 0;
}

extern "C" {

SP_LOWLEVEL_API void sp_median_uint8_nth_slab(
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

SP_LOWLEVEL_API void sp_median_uint16_nth_slab(
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

SP_LOWLEVEL_API void sp_median_int32_nth_slab(
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

SP_LOWLEVEL_API void sp_median_float32_nth_slab(
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

SP_LOWLEVEL_API void sp_convolve_float32_slices(
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
    convolve_scalar_slice_outputs<float>(slices, outputs, kernel, width, height, window_length, kernel_width, kernel_height, kernel_depth, output_start, output_count);
}

SP_LOWLEVEL_API int sp_signed_distance_band_uint8_slices(
    const uint8_t* const* slices,
    float* const* outputs,
    int width,
    int height,
    int window_length,
    int output_start,
    int output_count,
    float band_radius)
{
    return signed_distance_band_uint8_slices(
        slices,
        outputs,
        width,
        height,
        window_length,
        output_start,
        output_count,
        band_radius);
}

SP_LOWLEVEL_API void sp_convolve_uint8_slices(
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

#define SP_CONVOLVE_FIXED_EXPORT(TYPE, NAME) \
SP_LOWLEVEL_API void sp_convolve_##NAME##_slices( \
    const TYPE* const* slices, \
    TYPE* const* outputs, \
    const float* kernel, \
    int width, \
    int height, \
    int window_length, \
    int kernel_width, \
    int kernel_height, \
    int kernel_depth, \
    int output_start, \
    int output_count) \
{ \
    convolve_scalar_slice_outputs<TYPE>(slices, outputs, kernel, width, height, window_length, kernel_width, kernel_height, kernel_depth, output_start, output_count); \
}

SP_CONVOLVE_FIXED_EXPORT(int8_t, int8)
SP_CONVOLVE_FIXED_EXPORT(uint16_t, uint16)
SP_CONVOLVE_FIXED_EXPORT(int16_t, int16)
SP_CONVOLVE_FIXED_EXPORT(int32_t, int32)

#undef SP_CONVOLVE_FIXED_EXPORT

SP_LOWLEVEL_API void sp_convolve_uint8_x_slices(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    convolve_uint8_x_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        window_length,
        kernel_length,
        output_start,
        output_count);
}

SP_LOWLEVEL_API void sp_convolve_uint8_y_slices(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    convolve_uint8_y_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        window_length,
        kernel_length,
        output_start,
        output_count);
}

SP_LOWLEVEL_API void sp_convolve_uint8_z_slices(
    const uint8_t* const* slices,
    uint8_t* const* outputs,
    const float* kernel,
    int width,
    int height,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    convolve_uint8_z_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        window_length,
        kernel_length,
        output_start,
        output_count);
}

#define SP_CONVOLVE_AXIS_EXPORTS(TYPE, NAME) \
SP_LOWLEVEL_API void sp_convolve_##NAME##_x_slices( \
    const TYPE* const* slices, \
    TYPE* const* outputs, \
    const float* kernel, \
    int width, \
    int height, \
    int window_length, \
    int kernel_length, \
    int output_start, \
    int output_count) \
{ \
    convolve_axis_x_slice_outputs<TYPE>(slices, outputs, kernel, width, height, window_length, kernel_length, output_start, output_count); \
} \
SP_LOWLEVEL_API void sp_convolve_##NAME##_y_slices( \
    const TYPE* const* slices, \
    TYPE* const* outputs, \
    const float* kernel, \
    int width, \
    int height, \
    int window_length, \
    int kernel_length, \
    int output_start, \
    int output_count) \
{ \
    convolve_axis_y_slice_outputs<TYPE>(slices, outputs, kernel, width, height, window_length, kernel_length, output_start, output_count); \
} \
SP_LOWLEVEL_API void sp_convolve_##NAME##_z_slices( \
    const TYPE* const* slices, \
    TYPE* const* outputs, \
    const float* kernel, \
    int width, \
    int height, \
    int window_length, \
    int kernel_length, \
    int output_start, \
    int output_count) \
{ \
    convolve_axis_z_slice_outputs<TYPE>(slices, outputs, kernel, width, height, window_length, kernel_length, output_start, output_count); \
}

SP_CONVOLVE_AXIS_EXPORTS(int8_t, int8)
SP_CONVOLVE_AXIS_EXPORTS(uint16_t, uint16)
SP_CONVOLVE_AXIS_EXPORTS(int32_t, int32)
SP_CONVOLVE_AXIS_EXPORTS(float, float32)

#undef SP_CONVOLVE_AXIS_EXPORTS

SP_LOWLEVEL_API int sp_convolve_float32_vector_components_slices(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int components,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count,
    int axis)
{
    return convolve_float32_vector_components_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        components,
        window_length,
        kernel_length,
        output_start,
        output_count,
        axis);
}

SP_LOWLEVEL_API int sp_convolve_float32_vector_components_x_slices(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int components,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    return convolve_float32_vector_components_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        components,
        window_length,
        kernel_length,
        output_start,
        output_count,
        0);
}

SP_LOWLEVEL_API int sp_convolve_float32_vector_components_y_slices(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int components,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    return convolve_float32_vector_components_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        components,
        window_length,
        kernel_length,
        output_start,
        output_count,
        1);
}

SP_LOWLEVEL_API int sp_convolve_float32_vector_components_z_slices(
    const float* const* slices,
    float* const* outputs,
    const float* kernel,
    int width,
    int height,
    int components,
    int window_length,
    int kernel_length,
    int output_start,
    int output_count)
{
    return convolve_float32_vector_components_slice_outputs(
        slices,
        outputs,
        kernel,
        width,
        height,
        components,
        window_length,
        kernel_length,
        output_start,
        output_count,
        2);
}

}

enum SpPixelType {
    SP_PIXEL_UINT8 = 1,
    SP_PIXEL_INT8 = 2,
    SP_PIXEL_UINT16 = 3,
    SP_PIXEL_INT16 = 4,
    SP_PIXEL_INT32 = 5,
    SP_PIXEL_FLOAT32 = 6
};

enum SpInterpolation {
    SP_INTERP_NEAREST = 0,
    SP_INTERP_LINEAR = 1
};

template <typename T>
static inline T clamp_round_value(double value)
{
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(value);
    } else {
        if (std::isnan(value)) {
            return static_cast<T>(0);
        }
        const double lo = static_cast<double>(std::numeric_limits<T>::min());
        const double hi = static_cast<double>(std::numeric_limits<T>::max());
        if (value <= lo) {
            return std::numeric_limits<T>::min();
        }
        if (value >= hi) {
            return std::numeric_limits<T>::max();
        }
        return static_cast<T>(std::nearbyint(value));
    }
}

template <typename T>
static inline T sample_nearest_2d(
    const T* input,
    int width,
    int height,
    double x,
    double y)
{
    const int xi = static_cast<int>(std::floor(x + 0.5));
    const int yi = static_cast<int>(std::floor(y + 0.5));
    if (xi < 0 || yi < 0 || xi >= width || yi >= height) {
        return static_cast<T>(0);
    }
    return input[yi * width + xi];
}

template <typename T>
static inline T sample_linear_2d(
    const T* input,
    int width,
    int height,
    double x,
    double y)
{
    if (x < 0.0 || y < 0.0 || x > static_cast<double>(width - 1) || y > static_cast<double>(height - 1)) {
        return static_cast<T>(0);
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    const double c00 = static_cast<double>(input[y0 * width + x0]);
    const double c10 = static_cast<double>(input[y0 * width + x1]);
    const double c01 = static_cast<double>(input[y1 * width + x0]);
    const double c11 = static_cast<double>(input[y1 * width + x1]);
    const double c0 = c00 + (c10 - c00) * fx;
    const double c1 = c01 + (c11 - c01) * fx;
    return clamp_round_value<T>(c0 + (c1 - c0) * fy);
}

template <typename T>
static inline double sample_linear_2d_double(
    const T* input,
    int width,
    int height,
    double x,
    double y)
{
    if (input == nullptr || x < 0.0 || y < 0.0 || x > static_cast<double>(width - 1) || y > static_cast<double>(height - 1)) {
        return 0.0;
    }

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, width - 1);
    const int y1 = std::min(y0 + 1, height - 1);
    const double fx = x - static_cast<double>(x0);
    const double fy = y - static_cast<double>(y0);

    const double c00 = static_cast<double>(input[y0 * width + x0]);
    const double c10 = static_cast<double>(input[y0 * width + x1]);
    const double c01 = static_cast<double>(input[y1 * width + x0]);
    const double c11 = static_cast<double>(input[y1 * width + x1]);
    const double c0 = c00 + (c10 - c00) * fx;
    const double c1 = c01 + (c11 - c01) * fx;
    return c0 + (c1 - c0) * fy;
}

template <typename T>
static inline T sample_2d(
    const T* input,
    int width,
    int height,
    double x,
    double y,
    int interpolation)
{
    if (interpolation == SP_INTERP_NEAREST) {
        return sample_nearest_2d(input, width, height, x, y);
    }
    return sample_linear_2d(input, width, height, x, y);
}

template <typename T>
static int resample_2d_typed(
    const void* input_ptr,
    void* output_ptr,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    double spacing_x,
    double spacing_y,
    int interpolation)
{
    if (input_ptr == nullptr || output_ptr == nullptr || input_width <= 0 || input_height <= 0 || output_width <= 0 || output_height <= 0) {
        return 1;
    }
    if (spacing_x <= 0.0 || spacing_y <= 0.0) {
        return 2;
    }

    const T* input = static_cast<const T*>(input_ptr);
    T* output = static_cast<T*>(output_ptr);

    for (int y = 0; y < output_height; ++y) {
        const double source_y = static_cast<double>(y) * spacing_y;
        for (int x = 0; x < output_width; ++x) {
            const double source_x = static_cast<double>(x) * spacing_x;
            output[y * output_width + x] = sample_2d(input, input_width, input_height, source_x, source_y, interpolation);
        }
    }

    return 0;
}

template <typename T>
static int resize_3d_pair_slice_typed(
    const void* lower_ptr,
    const void* upper_ptr,
    void* output_ptr,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    double spacing_x,
    double spacing_y,
    double z_fraction,
    int interpolation)
{
    if (lower_ptr == nullptr || upper_ptr == nullptr || output_ptr == nullptr ||
        input_width <= 0 || input_height <= 0 || output_width <= 0 || output_height <= 0) {
        return 1;
    }
    if (spacing_x <= 0.0 || spacing_y <= 0.0 || z_fraction < 0.0 || z_fraction > 1.0) {
        return 2;
    }

    const T* lower = static_cast<const T*>(lower_ptr);
    const T* upper = static_cast<const T*>(upper_ptr);
    T* output = static_cast<T*>(output_ptr);

    if (interpolation == SP_INTERP_NEAREST) {
        const T* selected = z_fraction < 0.5 ? lower : upper;
        for (int y = 0; y < output_height; ++y) {
            const double source_y = static_cast<double>(y) * spacing_y;
            for (int x = 0; x < output_width; ++x) {
                const double source_x = static_cast<double>(x) * spacing_x;
                output[y * output_width + x] = sample_nearest_2d(selected, input_width, input_height, source_x, source_y);
            }
        }
    } else {
        for (int y = 0; y < output_height; ++y) {
            const double source_y = static_cast<double>(y) * spacing_y;
            for (int x = 0; x < output_width; ++x) {
                const double source_x = static_cast<double>(x) * spacing_x;
                const double lower_value = sample_linear_2d_double(lower, input_width, input_height, source_x, source_y);
                const double upper_value = sample_linear_2d_double(upper, input_width, input_height, source_x, source_y);
                output[y * output_width + x] = clamp_round_value<T>(lower_value + (upper_value - lower_value) * z_fraction);
            }
        }
    }

    return 0;
}

template <typename T>
static int euler_2d_typed(
    const void* input_ptr,
    void* output_ptr,
    int width,
    int height,
    double center_x,
    double center_y,
    double angle,
    double dx,
    double dy,
    int inverse,
    int interpolation)
{
    if (input_ptr == nullptr || output_ptr == nullptr || width <= 0 || height <= 0) {
        return 1;
    }

    const T* input = static_cast<const T*>(input_ptr);
    T* output = static_cast<T*>(output_ptr);
    const double c = std::cos(angle);
    const double s = std::sin(angle);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double source_x = 0.0;
            double source_y = 0.0;

            if (inverse != 0) {
                const double qx = static_cast<double>(x) - center_x - dx;
                const double qy = static_cast<double>(y) - center_y - dy;
                source_x = c * qx + s * qy + center_x;
                source_y = -s * qx + c * qy + center_y;
            } else {
                const double qx = static_cast<double>(x) - center_x;
                const double qy = static_cast<double>(y) - center_y;
                source_x = c * qx - s * qy + center_x + dx;
                source_y = s * qx + c * qy + center_y + dy;
            }

            output[y * width + x] = sample_2d(input, width, height, source_x, source_y, interpolation);
        }
    }

    return 0;
}

extern "C" {

SP_LOWLEVEL_API int sp_resample_2d(
    const void* input,
    void* output,
    int pixel_type,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    double spacing_x,
    double spacing_y,
    int interpolation)
{
    switch (pixel_type) {
        case SP_PIXEL_UINT8:
            return resample_2d_typed<uint8_t>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        case SP_PIXEL_INT8:
            return resample_2d_typed<int8_t>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        case SP_PIXEL_UINT16:
            return resample_2d_typed<uint16_t>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        case SP_PIXEL_INT16:
            return resample_2d_typed<int16_t>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        case SP_PIXEL_INT32:
            return resample_2d_typed<int32_t>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        case SP_PIXEL_FLOAT32:
            return resample_2d_typed<float>(input, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, interpolation);
        default:
            return 100;
    }
}

SP_LOWLEVEL_API int sp_resize_3d_pair_slice(
    const void* lower,
    const void* upper,
    void* output,
    int pixel_type,
    int input_width,
    int input_height,
    int output_width,
    int output_height,
    double spacing_x,
    double spacing_y,
    double z_fraction,
    int interpolation)
{
    switch (pixel_type) {
        case SP_PIXEL_UINT8:
            return resize_3d_pair_slice_typed<uint8_t>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        case SP_PIXEL_INT8:
            return resize_3d_pair_slice_typed<int8_t>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        case SP_PIXEL_UINT16:
            return resize_3d_pair_slice_typed<uint16_t>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        case SP_PIXEL_INT16:
            return resize_3d_pair_slice_typed<int16_t>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        case SP_PIXEL_INT32:
            return resize_3d_pair_slice_typed<int32_t>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        case SP_PIXEL_FLOAT32:
            return resize_3d_pair_slice_typed<float>(lower, upper, output, input_width, input_height, output_width, output_height, spacing_x, spacing_y, z_fraction, interpolation);
        default:
            return 100;
    }
}

SP_LOWLEVEL_API int sp_euler_2d(
    const void* input,
    void* output,
    int pixel_type,
    int width,
    int height,
    double center_x,
    double center_y,
    double angle,
    double dx,
    double dy,
    int inverse,
    int interpolation)
{
    switch (pixel_type) {
        case SP_PIXEL_UINT8:
            return euler_2d_typed<uint8_t>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        case SP_PIXEL_INT8:
            return euler_2d_typed<int8_t>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        case SP_PIXEL_UINT16:
            return euler_2d_typed<uint16_t>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        case SP_PIXEL_INT16:
            return euler_2d_typed<int16_t>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        case SP_PIXEL_INT32:
            return euler_2d_typed<int32_t>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        case SP_PIXEL_FLOAT32:
            return euler_2d_typed<float>(input, output, width, height, center_x, center_y, angle, dx, dy, inverse, interpolation);
        default:
            return 100;
    }
}

SP_LOWLEVEL_API int sp_fftwf_complex_xy_inplace(
    float* interleaved,
    int width,
    int height,
    int inverse)
{
    return fftwf_complex_xy_inplace(interleaved, width, height, inverse);
}

SP_LOWLEVEL_API int sp_inv_fftwf_complex_xy_inplace(
    float* interleaved,
    int width,
    int height)
{
    return fftwf_complex_xy_inplace(interleaved, width, height, 1);
}

SP_LOWLEVEL_API int sp_fftwf_complex_z_inplace(
    float* interleaved,
    int width,
    int height,
    int depth,
    int inverse)
{
    return fftwf_complex_z_inplace(interleaved, width, height, depth, inverse);
}

SP_LOWLEVEL_API int sp_inv_fftwf_complex_z_inplace(
    float* interleaved,
    int width,
    int height,
    int depth)
{
    return fftwf_complex_z_inplace(interleaved, width, height, depth, 1);
}

}
