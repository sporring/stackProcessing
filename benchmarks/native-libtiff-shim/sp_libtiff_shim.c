#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <tiffio.h>

typedef struct sp_tiff_info {
    uint32_t width;
    uint32_t height;
    uint32_t rows_per_strip;
    uint32_t strips;
    uint16_t bits_per_sample;
    uint16_t sample_format;
    uint16_t samples_per_pixel;
    uint16_t planar_config;
    uint16_t compression;
    int32_t is_tiled;
    int32_t is_byte_swapped;
    uint64_t page_bytes;
    uint64_t raw_page_bytes;
} sp_tiff_info;

enum {
    SP_TIFF_OK = 0,
    SP_TIFF_OPEN_FAILED = -1,
    SP_TIFF_MISSING_FIELD = -2,
    SP_TIFF_UNSUPPORTED_LAYOUT = -3,
    SP_TIFF_BUFFER_TOO_SMALL = -4,
    SP_TIFF_IO_FAILED = -5,
    SP_TIFF_SIZE_OVERFLOW = -6
};

static int read_info_from_open_tiff(TIFF *tif, sp_tiff_info *info) {
    uint16_t sample_format = SAMPLEFORMAT_UINT;
    uint16_t samples_per_pixel = 1;
    uint16_t planar_config = PLANARCONFIG_CONTIG;
    uint16_t compression = COMPRESSION_NONE;
    uint32_t rows_per_strip = 0;
    uint64_t raw_page_bytes = 0;

    memset(info, 0, sizeof(*info));

    if (!TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &info->width) ||
        !TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &info->height) ||
        !TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &info->bits_per_sample)) {
        return SP_TIFF_MISSING_FIELD;
    }

    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLESPERPIXEL, &samples_per_pixel);
    TIFFGetFieldDefaulted(tif, TIFFTAG_PLANARCONFIG, &planar_config);
    TIFFGetFieldDefaulted(tif, TIFFTAG_COMPRESSION, &compression);
    TIFFGetFieldDefaulted(tif, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);

    info->sample_format = sample_format;
    info->samples_per_pixel = samples_per_pixel;
    info->planar_config = planar_config;
    info->compression = compression;
    info->rows_per_strip = rows_per_strip;
    info->strips = TIFFNumberOfStrips(tif);
    info->is_tiled = TIFFIsTiled(tif);
    info->is_byte_swapped = TIFFIsByteSwapped(tif);

    if (info->width == 0 || info->height == 0 || info->bits_per_sample == 0 ||
        info->samples_per_pixel == 0 || info->bits_per_sample % 8 != 0 ||
        info->is_tiled || info->planar_config != PLANARCONFIG_CONTIG) {
        return SP_TIFF_UNSUPPORTED_LAYOUT;
    }

    uint64_t bytes_per_pixel = ((uint64_t)info->bits_per_sample / 8u) * (uint64_t)info->samples_per_pixel;
    if (bytes_per_pixel == 0 ||
        info->width > UINT64_MAX / bytes_per_pixel ||
        (uint64_t)info->width * bytes_per_pixel > UINT64_MAX / (uint64_t)info->height) {
        return SP_TIFF_SIZE_OVERFLOW;
    }

    info->page_bytes = (uint64_t)info->width * (uint64_t)info->height * bytes_per_pixel;

    for (uint32_t strip = 0; strip < info->strips; strip++) {
        uint64_t strip_size = TIFFRawStripSize64(tif, strip);
        if (strip_size == UINT64_MAX || raw_page_bytes > UINT64_MAX - strip_size) {
            return SP_TIFF_SIZE_OVERFLOW;
        }
        raw_page_bytes += strip_size;
    }

    info->raw_page_bytes = raw_page_bytes;
    return SP_TIFF_OK;
}

#if defined(_WIN32)
#define SP_EXPORT __declspec(dllexport)
#else
#define SP_EXPORT __attribute__((visibility("default")))
#endif

SP_EXPORT int sp_tiff_read_info(const char *path, sp_tiff_info *info) {
    if (path == NULL || info == NULL) {
        return SP_TIFF_MISSING_FIELD;
    }

    TIFF *tif = TIFFOpen(path, "r");
    if (tif == NULL) {
        return SP_TIFF_OPEN_FAILED;
    }

    int result = read_info_from_open_tiff(tif, info);
    TIFFClose(tif);
    return result;
}

SP_EXPORT int sp_tiff_read_raw_page_into(const char *path, uint8_t *buffer, size_t buffer_offset, size_t capacity, uint64_t *bytes_read) {
    sp_tiff_info info;
    uint64_t offset = 0;

    if (bytes_read != NULL) {
        *bytes_read = 0;
    }
    if (path == NULL || buffer == NULL || bytes_read == NULL) {
        return SP_TIFF_MISSING_FIELD;
    }

    TIFF *tif = TIFFOpen(path, "r");
    if (tif == NULL) {
        return SP_TIFF_OPEN_FAILED;
    }

    int result = read_info_from_open_tiff(tif, &info);
    if (result != SP_TIFF_OK) {
        TIFFClose(tif);
        return result;
    }
    if ((uint64_t)buffer_offset > (uint64_t)capacity ||
        (uint64_t)capacity - (uint64_t)buffer_offset < info.raw_page_bytes) {
        TIFFClose(tif);
        return SP_TIFF_BUFFER_TOO_SMALL;
    }

    for (uint32_t strip = 0; strip < info.strips; strip++) {
        uint64_t strip_size_64 = TIFFRawStripSize64(tif, strip);
        if (strip_size_64 > (uint64_t)((tmsize_t)-1)) {
            TIFFClose(tif);
            return SP_TIFF_SIZE_OVERFLOW;
        }

        tmsize_t strip_size = (tmsize_t)strip_size_64;
        tmsize_t got = TIFFReadRawStrip(tif, strip, buffer + buffer_offset + offset, strip_size);
        if (got != strip_size) {
            TIFFClose(tif);
            return SP_TIFF_IO_FAILED;
        }
        offset += strip_size_64;
    }

    *bytes_read = offset;
    TIFFClose(tif);
    return SP_TIFF_OK;
}

SP_EXPORT int sp_tiff_read_raw_page(const char *path, uint8_t *buffer, size_t capacity, uint64_t *bytes_read) {
    return sp_tiff_read_raw_page_into(path, buffer, 0, capacity, bytes_read);
}

SP_EXPORT int sp_tiff_read_scanline_page(const char *path, uint8_t *buffer, size_t capacity, uint64_t *bytes_read) {
    sp_tiff_info info;

    if (bytes_read != NULL) {
        *bytes_read = 0;
    }
    if (path == NULL || buffer == NULL || bytes_read == NULL) {
        return SP_TIFF_MISSING_FIELD;
    }

    TIFF *tif = TIFFOpen(path, "r");
    if (tif == NULL) {
        return SP_TIFF_OPEN_FAILED;
    }

    int result = read_info_from_open_tiff(tif, &info);
    if (result != SP_TIFF_OK) {
        TIFFClose(tif);
        return result;
    }
    if ((uint64_t)capacity < info.page_bytes || info.height == 0) {
        TIFFClose(tif);
        return SP_TIFF_BUFFER_TOO_SMALL;
    }

    const uint64_t row_bytes_64 = info.page_bytes / (uint64_t)info.height;
    if (row_bytes_64 > (uint64_t)((tmsize_t)-1)) {
        TIFFClose(tif);
        return SP_TIFF_SIZE_OVERFLOW;
    }

    const tmsize_t scanline_size = TIFFScanlineSize(tif);
    if (scanline_size < 0 || (uint64_t)scanline_size > row_bytes_64) {
        TIFFClose(tif);
        return SP_TIFF_UNSUPPORTED_LAYOUT;
    }

    size_t offset = 0;
    const size_t row_bytes = (size_t)row_bytes_64;
    for (uint32_t row = 0; row < info.height; row++) {
        if (TIFFReadScanline(tif, buffer + offset, row, 0) < 0) {
            TIFFClose(tif);
            return SP_TIFF_IO_FAILED;
        }
        offset += row_bytes;
    }

    *bytes_read = (uint64_t)offset;
    TIFFClose(tif);
    return SP_TIFF_OK;
}

SP_EXPORT int sp_tiff_write_raw_page(
    const char *path,
    const uint8_t *buffer,
    size_t count,
    uint32_t width,
    uint32_t height,
    uint16_t bits_per_sample,
    uint16_t sample_format) {

    if (path == NULL || buffer == NULL || width == 0 || height == 0 || bits_per_sample == 0) {
        return SP_TIFF_MISSING_FIELD;
    }
    if ((uint64_t)count > (uint64_t)((tmsize_t)-1)) {
        return SP_TIFF_SIZE_OVERFLOW;
    }

    TIFF *tif = TIFFOpen(path, "w");
    if (tif == NULL) {
        return SP_TIFF_OPEN_FAILED;
    }

    int ok =
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width) &&
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height) &&
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1) &&
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample) &&
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format) &&
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height) &&
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

    if (!ok) {
        TIFFClose(tif);
        return SP_TIFF_MISSING_FIELD;
    }

    tmsize_t written = TIFFWriteRawStrip(tif, 0, (void *)buffer, (tmsize_t)count);
    if (written != (tmsize_t)count) {
        TIFFClose(tif);
        return SP_TIFF_IO_FAILED;
    }

    TIFFClose(tif);
    return SP_TIFF_OK;
}

SP_EXPORT int sp_tiff_write_raw_page_from(
    const char *path,
    const uint8_t *buffer,
    size_t buffer_offset,
    size_t count,
    size_t capacity,
    uint32_t width,
    uint32_t height,
    uint16_t bits_per_sample,
    uint16_t sample_format) {

    if (buffer_offset > capacity || capacity - buffer_offset < count) {
        return SP_TIFF_BUFFER_TOO_SMALL;
    }

    return sp_tiff_write_raw_page(
        path,
        buffer + buffer_offset,
        count,
        width,
        height,
        bits_per_sample,
        sample_format);
}

SP_EXPORT int sp_tiff_write_scanline_page(
    const char *path,
    const uint8_t *buffer,
    size_t count,
    uint32_t width,
    uint32_t height,
    uint16_t bits_per_sample,
    uint16_t sample_format) {

    if (path == NULL || buffer == NULL || width == 0 || height == 0 || bits_per_sample == 0) {
        return SP_TIFF_MISSING_FIELD;
    }

    uint64_t bytes_per_pixel = ((uint64_t)bits_per_sample / 8u);
    if (bits_per_sample % 8 != 0 || bytes_per_pixel == 0 ||
        width > UINT64_MAX / bytes_per_pixel ||
        (uint64_t)width * bytes_per_pixel > UINT64_MAX / (uint64_t)height) {
        return SP_TIFF_SIZE_OVERFLOW;
    }

    const uint64_t row_bytes_64 = (uint64_t)width * bytes_per_pixel;
    const uint64_t page_bytes_64 = row_bytes_64 * (uint64_t)height;
    if ((uint64_t)count < page_bytes_64 || row_bytes_64 > (uint64_t)((tmsize_t)-1)) {
        return SP_TIFF_BUFFER_TOO_SMALL;
    }

    TIFF *tif = TIFFOpen(path, "w");
    if (tif == NULL) {
        return SP_TIFF_OPEN_FAILED;
    }

    int ok =
        TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width) &&
        TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height) &&
        TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1) &&
        TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, bits_per_sample) &&
        TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, sample_format) &&
        TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK) &&
        TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG) &&
        TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, height) &&
        TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);

    if (!ok) {
        TIFFClose(tif);
        return SP_TIFF_MISSING_FIELD;
    }

    size_t offset = 0;
    const tmsize_t row_bytes = (tmsize_t)row_bytes_64;
    for (uint32_t row = 0; row < height; row++) {
        if (TIFFWriteScanline(tif, (void *)(buffer + offset), row, 0) < 0) {
            TIFFClose(tif);
            return SP_TIFF_IO_FAILED;
        }
        offset += (size_t)row_bytes;
    }

    TIFFClose(tif);
    return SP_TIFF_OK;
}
