#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include "fftw3.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkConstantBoundaryCondition.h"
#include "itkConvolutionImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkForwardFFTImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkInverseFFTImageFilter.h"
#include "itkMedianImageFilter.h"
#include "itkTIFFImageIO.h"

namespace fs = std::filesystem;

static void writeInternalSeconds(std::chrono::duration<double> elapsed);

struct Options {
  std::string operation;
  std::string pixelType;
  fs::path input;
  fs::path output;
  std::string shape = "256x256x256";
  unsigned radius = 1;
  double threshold = 128.0;
  unsigned kernelSize = 3;
  unsigned window = 16;
  unsigned chunkSize = 64;
  unsigned iterations = 1;
};

static std::string argValue(int argc, char** argv, const std::string& name, const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

static Options parseOptions(int argc, char** argv) {
  Options options;
  options.operation = argValue(argc, argv, "--operation");
  options.pixelType = argValue(argc, argv, "--pixel-type");
  options.input = argValue(argc, argv, "--input");
  options.output = argValue(argc, argv, "--output");
  options.shape = argValue(argc, argv, "--shape", "256x256x256");
  options.radius = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--radius", "1")));
  options.threshold = std::stod(argValue(argc, argv, "--threshold", "128"));
  options.kernelSize = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--kernel-size", "3")));
  options.window = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--window", "16")));
  options.chunkSize = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--chunk-size", "64")));
  options.iterations = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--iterations", "1")));
  if (options.operation.empty() || options.pixelType.empty()) {
    throw std::runtime_error("required arguments: --operation --pixel-type");
  }
  if (options.operation != "thresholdKernel" &&
      options.operation != "thresholdKernelInType" &&
      options.operation != "fftKernel" &&
      options.operation != "fftRoundtripKernel" &&
      options.operation != "fft-roundtrip-kernel" &&
      (options.input.empty() || options.output.empty())) {
    throw std::runtime_error("required arguments for IO operations: --input --output");
  }
  return options;
}

static void writeTextFile(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("failed to open '" + path.string() + "' for writing");
  }
  out << text;
}

static bool isKernelOnlyOperation(const std::string& operation) {
  return operation == "thresholdKernel" ||
         operation == "thresholdKernelInType" ||
         operation == "fftKernel" ||
         operation == "fftRoundtripKernel" ||
         operation == "fft-roundtrip-kernel";
}

static bool isPrecleanedOutput(const fs::path& path) {
  const char* value = std::getenv("BENCHMARK_PRECLEANED_OUTPUTS");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }

  const auto target = fs::weakly_canonical(path);
#ifdef _WIN32
  constexpr char separator = ';';
#else
  constexpr char separator = ':';
#endif
  std::string paths(value);
  std::size_t start = 0;
  while (start <= paths.size()) {
    const auto end = paths.find(separator, start);
    const auto token = paths.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!token.empty() && fs::weakly_canonical(fs::path(token)) == target) {
      return true;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return false;
}

static void prepareOutput(const Options& options) {
  if (isKernelOnlyOperation(options.operation)) {
    return;
  }

  if (isPrecleanedOutput(options.output)) {
    fs::create_directories(options.output);
    return;
  }

  if (options.operation == "fft-zarr") {
    if (fs::exists(options.output)) {
      fs::remove_all(options.output);
    }
    fs::create_directories(options.output);
    return;
  }

  fs::create_directories(options.output);
  for (const auto& entry : fs::directory_iterator(options.output)) {
    if (entry.is_regular_file()) {
      fs::remove(entry.path());
    }
  }
}

template <typename ComplexImage>
static void writeComplex64Zarr(typename ComplexImage::Pointer image, const fs::path& output, unsigned chunkSize) {
  image->Update();

  if (chunkSize == 0) {
    throw std::runtime_error("--chunk-size must be positive");
  }

  fs::create_directories(output / "0" / "c");

  const auto region = image->GetLargestPossibleRegion();
  const auto size = region.GetSize();
  const auto width = static_cast<unsigned>(size[0]);
  const auto height = static_cast<unsigned>(size[1]);
  const auto depth = static_cast<unsigned>(size[2]);

  const std::string rootJson =
      "{\n"
      "  \"attributes\": {\n"
      "    \"multiscales\": [\n"
      "      {\n"
      "        \"version\": \"0.4\",\n"
      "        \"axes\": [\n"
      "          { \"name\": \"t\", \"type\": \"time\" },\n"
      "          { \"name\": \"c\", \"type\": \"channel\" },\n"
      "          { \"name\": \"z\", \"type\": \"space\" },\n"
      "          { \"name\": \"y\", \"type\": \"space\" },\n"
      "          { \"name\": \"x\", \"type\": \"space\" }\n"
      "        ],\n"
      "        \"datasets\": [ { \"path\": \"0\" } ]\n"
      "      }\n"
      "    ],\n"
      "    \"omero\": { \"channels\": [ { \"label\": \"0\" } ] }\n"
      "  },\n"
      "  \"zarr_format\": 3,\n"
      "  \"node_type\": \"group\"\n"
      "}\n";

  const std::string arrayJson =
      "{\n"
      "  \"shape\": [ 1, 1, " + std::to_string(depth) + ", " + std::to_string(height) + ", " + std::to_string(width) + " ],\n"
      "  \"data_type\": \"complex64\",\n"
      "  \"chunk_grid\": {\n"
      "    \"name\": \"regular\",\n"
      "    \"configuration\": { \"chunk_shape\": [ 1, 1, " + std::to_string(chunkSize) + ", " + std::to_string(chunkSize) + ", " + std::to_string(chunkSize) + " ] }\n"
      "  },\n"
      "  \"chunk_key_encoding\": { \"name\": \"default\", \"configuration\": { \"separator\": \"/\" } },\n"
      "  \"fill_value\": [ 0.0, 0.0 ],\n"
      "  \"codecs\": [ { \"name\": \"bytes\", \"configuration\": { \"endian\": \"little\" } } ],\n"
      "  \"attributes\": {},\n"
      "  \"zarr_format\": 3,\n"
      "  \"node_type\": \"array\",\n"
      "  \"storage_transformers\": []\n"
      "}\n";

  writeTextFile(output / "zarr.json", rootJson);
  writeTextFile(output / "0" / "zarr.json", arrayJson);

  const auto* buffer = image->GetBufferPointer();
  const auto xChunks = (width + chunkSize - 1) / chunkSize;
  const auto yChunks = (height + chunkSize - 1) / chunkSize;
  const auto zChunks = (depth + chunkSize - 1) / chunkSize;

  for (unsigned zc = 0; zc < zChunks; ++zc) {
    const auto z0 = zc * chunkSize;
    const auto z1 = std::min(depth, z0 + chunkSize);
    for (unsigned yc = 0; yc < yChunks; ++yc) {
      const auto y0 = yc * chunkSize;
      const auto y1 = std::min(height, y0 + chunkSize);
      for (unsigned xc = 0; xc < xChunks; ++xc) {
        const auto x0 = xc * chunkSize;
        const auto x1 = std::min(width, x0 + chunkSize);

        const auto values =
            static_cast<std::size_t>(z1 - z0) * static_cast<std::size_t>(y1 - y0) * static_cast<std::size_t>(x1 - x0);
        std::vector<float> chunk;
        chunk.reserve(values * 2u);

        for (unsigned z = z0; z < z1; ++z) {
          for (unsigned y = y0; y < y1; ++y) {
            const auto row = (static_cast<std::size_t>(z) * height + y) * width;
            for (unsigned x = x0; x < x1; ++x) {
              const auto value = buffer[row + x];
              chunk.push_back(static_cast<float>(value.real()));
              chunk.push_back(static_cast<float>(value.imag()));
            }
          }
        }

        const auto chunkPath =
            output / "0" / "c" / "0" / "0" / std::to_string(zc) / std::to_string(yc) / std::to_string(xc);
        fs::create_directories(chunkPath.parent_path());
        std::ofstream out(chunkPath, std::ios::binary);
        if (!out) {
          throw std::runtime_error("failed to open chunk '" + chunkPath.string() + "' for writing");
        }
        out.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size() * sizeof(float)));
      }
    }
  }
}

static std::array<unsigned, 3> parseShape(const std::string& shape) {
  std::array<unsigned, 3> result{};
  std::size_t start = 0;
  for (std::size_t i = 0; i < 3; ++i) {
    const auto end = (i == 2) ? std::string::npos : shape.find('x', start);
    if (end == std::string::npos && i != 2) {
      throw std::runtime_error("shape must be WxHxD");
    }
    result[i] = static_cast<unsigned>(std::stoul(shape.substr(start, end - start)));
    start = end + 1;
  }
  return result;
}

template <typename T>
static T binaryMaskLowerThreshold() {
  return static_cast<T>(128);
}

static std::vector<fs::path> tiffFiles(const fs::path& input) {
  std::vector<fs::path> paths;
  for (const auto& entry : fs::directory_iterator(input)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    auto ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    if (ext == ".tif" || ext == ".tiff") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    throw std::runtime_error("no TIFF files found in input directory");
  }
  return paths;
}

template <typename T>
static typename itk::Image<T, 3>::Pointer readVolume(const std::vector<fs::path>& paths) {
  using Image2 = itk::Image<T, 2>;
  using Image3 = itk::Image<T, 3>;
  using Reader = itk::ImageFileReader<Image2>;
  using ImageIO = itk::TIFFImageIO;

  auto firstReader = Reader::New();
  firstReader->SetImageIO(ImageIO::New());
  firstReader->SetFileName(paths.front().string());
  firstReader->Update();
  auto first = firstReader->GetOutput();
  const auto sliceRegion = first->GetLargestPossibleRegion();
  const auto sliceSize = sliceRegion.GetSize();

  typename Image3::SizeType volumeSize;
  volumeSize[0] = sliceSize[0];
  volumeSize[1] = sliceSize[1];
  volumeSize[2] = paths.size();

  typename Image3::IndexType volumeStart;
  volumeStart.Fill(0);
  typename Image3::RegionType volumeRegion;
  volumeRegion.SetIndex(volumeStart);
  volumeRegion.SetSize(volumeSize);

  auto volume = Image3::New();
  volume->SetRegions(volumeRegion);
  volume->Allocate();

  const auto pixelsPerSlice = static_cast<std::size_t>(sliceSize[0] * sliceSize[1]);
  T* volumeBuffer = volume->GetBufferPointer();
  const T* firstBuffer = first->GetBufferPointer();
  std::copy(firstBuffer, firstBuffer + pixelsPerSlice, volumeBuffer);

  for (std::size_t z = 1; z < paths.size(); ++z) {
    auto reader = Reader::New();
    reader->SetImageIO(ImageIO::New());
    reader->SetFileName(paths[z].string());
    reader->Update();
    auto slice = reader->GetOutput();

    if (slice->GetLargestPossibleRegion().GetSize() != sliceSize) {
      throw std::runtime_error("all TIFF slices must have the same size");
    }

    const T* sliceBuffer = slice->GetBufferPointer();
    std::copy(sliceBuffer, sliceBuffer + pixelsPerSlice, volumeBuffer + z * pixelsPerSlice);
  }

  return volume;
}

template <typename T>
static typename itk::Image<T, 3>::Pointer makeKernelVolume(const std::string& shapeText) {
  using Image3 = itk::Image<T, 3>;
  const auto parsed = parseShape(shapeText);

  typename Image3::SizeType volumeSize;
  volumeSize[0] = parsed[0];
  volumeSize[1] = parsed[1];
  volumeSize[2] = parsed[2];

  typename Image3::IndexType volumeStart;
  volumeStart.Fill(0);
  typename Image3::RegionType volumeRegion;
  volumeRegion.SetIndex(volumeStart);
  volumeRegion.SetSize(volumeSize);

  auto volume = Image3::New();
  volume->SetRegions(volumeRegion);
  volume->Allocate();

  const auto pixels = static_cast<std::size_t>(volumeSize[0]) * static_cast<std::size_t>(volumeSize[1]) * static_cast<std::size_t>(volumeSize[2]);
  T* buffer = volume->GetBufferPointer();
  for (std::size_t i = 0; i < pixels; ++i) {
    buffer[i] = static_cast<T>(i & 0xFFu);
  }

  return volume;
}

template <typename Image3>
static void writeVolumeSlices(typename Image3::Pointer volume, const fs::path& output, const std::vector<fs::path>& inputNames) {
  using Image2 = itk::Image<typename Image3::PixelType, 2>;
  using Writer = itk::ImageFileWriter<Image2>;
  using ImageIO = itk::TIFFImageIO;

  volume->Update();

  const auto region = volume->GetLargestPossibleRegion();
  const auto size3 = region.GetSize();
  const auto depth = size3[2];
  const auto pixelsPerSlice = static_cast<std::size_t>(size3[0] * size3[1]);
  const auto* volumeBuffer = volume->GetBufferPointer();

  typename Image2::IndexType sliceStart;
  sliceStart.Fill(0);
  typename Image2::SizeType sliceSize;
  sliceSize[0] = size3[0];
  sliceSize[1] = size3[1];
  typename Image2::RegionType sliceRegion;
  sliceRegion.SetIndex(sliceStart);
  sliceRegion.SetSize(sliceSize);

  for (typename Image3::SizeValueType z = 0; z < depth; ++z) {
    auto slice = Image2::New();
    slice->SetRegions(sliceRegion);
    slice->Allocate();

    auto* sliceBuffer = slice->GetBufferPointer();
    const auto* source = volumeBuffer + static_cast<std::size_t>(z) * pixelsPerSlice;
    std::copy(source, source + pixelsPerSlice, sliceBuffer);

    auto writer = Writer::New();
    writer->SetImageIO(ImageIO::New());
    writer->SetFileName((output / inputNames[static_cast<std::size_t>(z)].filename()).string());
    writer->SetInput(slice);
    writer->UseCompressionOff();
    writer->Update();
  }
}

static itk::Image<double, 3>::Pointer makeUniformKernel(unsigned kernelSize) {
  using Kernel = itk::Image<double, 3>;
  const auto sizeValue = static_cast<Kernel::SizeValueType>(std::max(1u, kernelSize));
  Kernel::SizeType size;
  size.Fill(sizeValue);
  Kernel::IndexType start;
  start.Fill(0);
  Kernel::RegionType region;
  region.SetIndex(start);
  region.SetSize(size);

  auto kernel = Kernel::New();
  kernel->SetRegions(region);
  kernel->Allocate();
  kernel->FillBuffer(1.0 / static_cast<double>(sizeValue * sizeValue * sizeValue));
  return kernel;
}

static std::uint32_t checksumFloat(float value, std::uint32_t checksum) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return (checksum * 16777619u) ^ bits;
}

template <typename ComplexImage>
static std::uint32_t checksumComplexImage(typename ComplexImage::Pointer image) {
  image->Update();
  const auto region = image->GetLargestPossibleRegion();
  const auto size = region.GetSize();
  const auto width = static_cast<std::size_t>(size[0]);
  const auto height = static_cast<std::size_t>(size[1]);
  const auto depth = static_cast<std::size_t>(size[2]);
  const auto total = width * height * depth;
  const auto stride = std::max<std::size_t>(1, total / 4096);
  const auto* buffer = image->GetBufferPointer();
  std::uint32_t checksum = 2166136261u;
  for (std::size_t i = 0; i < total; i += stride) {
    checksum = checksumFloat(static_cast<float>(buffer[i].real()), checksum);
    checksum = checksumFloat(static_cast<float>(buffer[i].imag()), checksum);
  }
  checksum = (checksum * 16777619u) ^ static_cast<std::uint32_t>(total);
  return checksum;
}

static std::uint32_t checksumInterleavedComplex64(const std::vector<float>& values) {
  const auto complexCount = values.size() / 2u;
  const auto stride = std::max<std::size_t>(1, complexCount / 4096);
  std::uint32_t checksum = 2166136261u;
  for (std::size_t i = 0; i < complexCount; i += stride) {
    checksum = checksumFloat(values[2u * i], checksum);
    checksum = checksumFloat(values[2u * i + 1u], checksum);
  }
  checksum = (checksum * 16777619u) ^ static_cast<std::uint32_t>(complexCount);
  return checksum;
}

static int fftwfComplexXYInplace(float* interleaved, int width, int height) {
  fftwf_complex* data = reinterpret_cast<fftwf_complex*>(interleaved);
  fftwf_plan plan = fftwf_plan_dft_2d(height, width, data, data, FFTW_FORWARD, FFTW_ESTIMATE);
  if (plan == nullptr) {
    return 2;
  }
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  return 0;
}

static int fftwfComplexZInplace(float* interleaved, int width, int height, int depth) {
  const int plane = width * height;
  fftwf_complex* data = reinterpret_cast<fftwf_complex*>(interleaved);
  int n[] = { depth };
  fftwf_plan plan =
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
          FFTW_FORWARD,
          FFTW_ESTIMATE);
  if (plan == nullptr) {
    return 2;
  }
  fftwf_execute(plan);
  fftwf_destroy_plan(plan);
  return 0;
}

static void fillInterleavedComplex64FromReal(const float* source, std::vector<float>& values) {
  const auto count = values.size() / 2u;
  for (std::size_t i = 0; i < count; ++i) {
    values[2u * i] = source[i];
    values[2u * i + 1u] = 0.0f;
  }
}

static std::uint32_t checksumFloat32Buffer(const float* values, std::size_t count) {
  const auto stride = std::max<std::size_t>(1, count / 4096);
  std::uint32_t checksum = 2166136261u;
  for (std::size_t i = 0; i < count; i += stride) {
    checksum = checksumFloat(values[i], checksum);
  }
  checksum = (checksum * 16777619u) ^ static_cast<std::uint32_t>(count);
  return checksum;
}

template <typename FloatImage>
static std::uint32_t checksumFloatImage(typename FloatImage::Pointer image) {
  image->Update();
  const auto region = image->GetLargestPossibleRegion();
  const auto size = region.GetSize();
  const auto total =
      static_cast<std::size_t>(size[0]) *
      static_cast<std::size_t>(size[1]) *
      static_cast<std::size_t>(size[2]);
  return checksumFloat32Buffer(image->GetBufferPointer(), total);
}

static int fftwfRealXYToComplexPlanExecute(
    fftwf_plan plan,
    float* real,
    float* interleaved) {
  if (plan == nullptr || real == nullptr || interleaved == nullptr) {
    return 1;
  }
  auto* output = reinterpret_cast<fftwf_complex*>(interleaved);
  fftwf_execute_dft_r2c(plan, real, output);
  return 0;
}

static int fftwfComplexXYToRealPlanExecute(
    fftwf_plan plan,
    float* interleaved,
    float* real,
    int width,
    int height) {
  if (plan == nullptr || interleaved == nullptr || real == nullptr) {
    return 1;
  }
  auto* input = reinterpret_cast<fftwf_complex*>(interleaved);
  fftwf_execute_dft_c2r(plan, input, real);
  const float scale = 1.0f / static_cast<float>(width * height);
  const int count = width * height;
  for (int i = 0; i < count; ++i) {
    real[i] *= scale;
  }
  return 0;
}

static int fftwfComplexZPlanExecute(
    fftwf_plan plan,
    float* interleaved,
    int complexWidth,
    int height,
    int depth,
    bool inverse) {
  if (plan == nullptr || interleaved == nullptr) {
    return 1;
  }
  auto* data = reinterpret_cast<fftwf_complex*>(interleaved);
  fftwf_execute_dft(plan, data, data);
  if (inverse) {
    const float scale = 1.0f / static_cast<float>(depth);
    const std::size_t complexCount =
        static_cast<std::size_t>(complexWidth) *
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(depth);
    for (std::size_t i = 0; i < complexCount * 2u; ++i) {
      interleaved[i] *= scale;
    }
  }
  return 0;
}

static void runFftRoundtripKernel(const Options& options) {
  if (options.pixelType != "Float32") {
    throw std::runtime_error("fftRoundtripKernel currently expects --pixel-type Float32");
  }
  if (options.iterations == 0) {
    throw std::runtime_error("--iterations must be positive");
  }

  using FloatImage = itk::Image<float, 3>;
  using FFT = itk::ForwardFFTImageFilter<FloatImage>;
  using ComplexImage = typename FFT::OutputImageType;
  using IFFT = itk::InverseFFTImageFilter<ComplexImage, FloatImage>;

  auto input = makeKernelVolume<float>(options.shape);
  input->Update();
  const auto parsed = parseShape(options.shape);
  const auto width = static_cast<int>(parsed[0]);
  const auto height = static_cast<int>(parsed[1]);
  const auto depth = static_cast<int>(parsed[2]);
  const auto realPlane =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const auto total = realPlane * static_cast<std::size_t>(depth);
  const auto complexWidth = width / 2 + 1;
  const auto complexPlane =
      static_cast<std::size_t>(complexWidth) * static_cast<std::size_t>(height);
  const auto complexCount = complexPlane * static_cast<std::size_t>(depth);
  const float* source = input->GetBufferPointer();

  typename FloatImage::Pointer itkOutput;
  auto itkStart = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < options.iterations; ++i) {
    auto fft = FFT::New();
    fft->SetInput(input);
    auto ifft = IFFT::New();
    ifft->SetInput(fft->GetOutput());
    ifft->Update();
    itkOutput = ifft->GetOutput();
    itkOutput->DisconnectPipeline();
  }
  const std::chrono::duration<double> itkElapsed = std::chrono::steady_clock::now() - itkStart;
  const auto itkChecksum = checksumFloatImage<FloatImage>(itkOutput);

  std::vector<float> real(total);
  std::vector<float> output(total);
  std::vector<float> complexValues(complexCount * 2u);

  std::memcpy(real.data(), source, total * sizeof(float));
  fftwf_plan xyForwardPlan = fftwf_plan_dft_r2c_2d(
      height,
      width,
      real.data(),
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      FFTW_ESTIMATE);
  fftwf_plan zForwardPlan = nullptr;
  fftwf_plan zInversePlan = nullptr;
  fftwf_plan xyInversePlan = nullptr;
  if (xyForwardPlan == nullptr) {
    throw std::runtime_error("lowlevel r2c xy plan creation failed");
  }
  int zN[] = { depth };
  zForwardPlan = fftwf_plan_many_dft(
      1,
      zN,
      complexWidth * height,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      FFTW_FORWARD,
      FFTW_ESTIMATE);
  zInversePlan = fftwf_plan_many_dft(
      1,
      zN,
      complexWidth * height,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      FFTW_BACKWARD,
      FFTW_ESTIMATE);
  xyInversePlan = fftwf_plan_dft_c2r_2d(
      height,
      width,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      output.data(),
      FFTW_ESTIMATE);
  if (zForwardPlan == nullptr || zInversePlan == nullptr || xyInversePlan == nullptr) {
    fftwf_destroy_plan(xyForwardPlan);
    if (zForwardPlan != nullptr) fftwf_destroy_plan(zForwardPlan);
    if (zInversePlan != nullptr) fftwf_destroy_plan(zInversePlan);
    if (xyInversePlan != nullptr) fftwf_destroy_plan(xyInversePlan);
    throw std::runtime_error("lowlevel z/xy inverse plan creation failed");
  }

  std::chrono::duration<double> lowlevelElapsed{};
  try {
    for (unsigned i = 0; i < options.iterations; ++i) {
      const auto start = std::chrono::steady_clock::now();
      std::memcpy(real.data(), source, total * sizeof(float));
      for (int z = 0; z < depth; ++z) {
        const auto realOffset = static_cast<std::size_t>(z) * realPlane;
        const auto complexOffset = static_cast<std::size_t>(z) * complexPlane * 2u;
        const int status =
            fftwfRealXYToComplexPlanExecute(
                xyForwardPlan,
                real.data() + realOffset,
                complexValues.data() + complexOffset);
        if (status != 0) {
          throw std::runtime_error("lowlevel r2c fftxy failed");
        }
      }
      if (fftwfComplexZPlanExecute(zForwardPlan, complexValues.data(), complexWidth, height, depth, false) != 0) {
        throw std::runtime_error("lowlevel c2c fftz failed");
      }
      if (fftwfComplexZPlanExecute(zInversePlan, complexValues.data(), complexWidth, height, depth, true) != 0) {
        throw std::runtime_error("lowlevel c2c invfftz failed");
      }
      for (int z = 0; z < depth; ++z) {
        const auto complexOffset = static_cast<std::size_t>(z) * complexPlane * 2u;
        const auto realOffset = static_cast<std::size_t>(z) * realPlane;
        const int status =
            fftwfComplexXYToRealPlanExecute(
                xyInversePlan,
                complexValues.data() + complexOffset,
                output.data() + realOffset,
                width,
                height);
        if (status != 0) {
          throw std::runtime_error("lowlevel c2r invfftxy failed");
        }
      }
      lowlevelElapsed += std::chrono::steady_clock::now() - start;
    }
  } catch (...) {
    fftwf_destroy_plan(xyForwardPlan);
    fftwf_destroy_plan(zForwardPlan);
    fftwf_destroy_plan(zInversePlan);
    fftwf_destroy_plan(xyInversePlan);
    throw;
  }
  fftwf_destroy_plan(xyForwardPlan);
  fftwf_destroy_plan(zForwardPlan);
  fftwf_destroy_plan(zInversePlan);
  fftwf_destroy_plan(xyInversePlan);

  const auto lowlevelChecksum = checksumFloat32Buffer(output.data(), output.size());

  std::cout.precision(9);
  std::cout << std::fixed
            << "variant=cpp-itk-fft-roundtrip shape=" << options.shape
            << " iterations=" << options.iterations
            << " totalSeconds=" << itkElapsed.count()
            << " perIterationSeconds=" << (itkElapsed.count() / static_cast<double>(options.iterations))
            << " checksum=" << itkChecksum << "\n"
            << "variant=lowlevel-r2c-xy-c2c-z-roundtrip shape=" << options.shape
            << " iterations=" << options.iterations
            << " totalSeconds=" << lowlevelElapsed.count()
            << " perIterationSeconds=" << (lowlevelElapsed.count() / static_cast<double>(options.iterations))
            << " checksum=" << lowlevelChecksum << "\n";
  writeInternalSeconds(itkElapsed + lowlevelElapsed);
}

static void runLowlevelFftRoundtripIo(
    typename itk::Image<float, 3>::Pointer input,
    const fs::path& outputPath,
    const std::vector<fs::path>& inputNames) {
  using FloatImage = itk::Image<float, 3>;

  input->Update();
  const auto region = input->GetLargestPossibleRegion();
  const auto size = region.GetSize();
  const int width = static_cast<int>(size[0]);
  const int height = static_cast<int>(size[1]);
  const int depth = static_cast<int>(size[2]);
  const auto realPlane =
      static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const auto total = realPlane * static_cast<std::size_t>(depth);
  const int complexWidth = width / 2 + 1;
  const auto complexPlane =
      static_cast<std::size_t>(complexWidth) * static_cast<std::size_t>(height);
  const auto complexCount = complexPlane * static_cast<std::size_t>(depth);

  std::vector<float> real(total);
  std::vector<float> recovered(total);
  std::vector<float> complexValues(complexCount * 2u);
  std::memcpy(real.data(), input->GetBufferPointer(), total * sizeof(float));

  fftwf_plan xyForwardPlan = fftwf_plan_dft_r2c_2d(
      height,
      width,
      real.data(),
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      FFTW_ESTIMATE);
  int zN[] = { depth };
  fftwf_plan zForwardPlan = fftwf_plan_many_dft(
      1,
      zN,
      complexWidth * height,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      FFTW_FORWARD,
      FFTW_ESTIMATE);
  fftwf_plan zInversePlan = fftwf_plan_many_dft(
      1,
      zN,
      complexWidth * height,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      nullptr,
      complexWidth * height,
      1,
      FFTW_BACKWARD,
      FFTW_ESTIMATE);
  fftwf_plan xyInversePlan = fftwf_plan_dft_c2r_2d(
      height,
      width,
      reinterpret_cast<fftwf_complex*>(complexValues.data()),
      recovered.data(),
      FFTW_ESTIMATE);

  if (xyForwardPlan == nullptr || zForwardPlan == nullptr || zInversePlan == nullptr || xyInversePlan == nullptr) {
    if (xyForwardPlan != nullptr) fftwf_destroy_plan(xyForwardPlan);
    if (zForwardPlan != nullptr) fftwf_destroy_plan(zForwardPlan);
    if (zInversePlan != nullptr) fftwf_destroy_plan(zInversePlan);
    if (xyInversePlan != nullptr) fftwf_destroy_plan(xyInversePlan);
    throw std::runtime_error("lowlevel FFTW IO roundtrip plan creation failed");
  }

  try {
    for (int z = 0; z < depth; ++z) {
      const auto realOffset = static_cast<std::size_t>(z) * realPlane;
      const auto complexOffset = static_cast<std::size_t>(z) * complexPlane * 2u;
      const int status =
          fftwfRealXYToComplexPlanExecute(
              xyForwardPlan,
              real.data() + realOffset,
              complexValues.data() + complexOffset);
      if (status != 0) {
        throw std::runtime_error("lowlevel IO r2c fftxy failed");
      }
    }
    if (fftwfComplexZPlanExecute(zForwardPlan, complexValues.data(), complexWidth, height, depth, false) != 0) {
      throw std::runtime_error("lowlevel IO c2c fftz failed");
    }
    if (fftwfComplexZPlanExecute(zInversePlan, complexValues.data(), complexWidth, height, depth, true) != 0) {
      throw std::runtime_error("lowlevel IO c2c invfftz failed");
    }
    for (int z = 0; z < depth; ++z) {
      const auto complexOffset = static_cast<std::size_t>(z) * complexPlane * 2u;
      const auto realOffset = static_cast<std::size_t>(z) * realPlane;
      const int status =
          fftwfComplexXYToRealPlanExecute(
              xyInversePlan,
              complexValues.data() + complexOffset,
              recovered.data() + realOffset,
              width,
              height);
      if (status != 0) {
        throw std::runtime_error("lowlevel IO c2r invfftxy failed");
      }
    }
  } catch (...) {
    fftwf_destroy_plan(xyForwardPlan);
    fftwf_destroy_plan(zForwardPlan);
    fftwf_destroy_plan(zInversePlan);
    fftwf_destroy_plan(xyInversePlan);
    throw;
  }

  fftwf_destroy_plan(xyForwardPlan);
  fftwf_destroy_plan(zForwardPlan);
  fftwf_destroy_plan(zInversePlan);
  fftwf_destroy_plan(xyInversePlan);

  auto output = FloatImage::New();
  output->SetRegions(region);
  output->Allocate();
  std::memcpy(output->GetBufferPointer(), recovered.data(), total * sizeof(float));

  const auto checksum = checksumFloat32Buffer(recovered.data(), recovered.size());
  std::cout.precision(9);
  std::cout << std::fixed
            << "variant=lowlevel-r2c-xy-c2c-z-roundtrip-io"
            << " shape=" << width << "x" << height << "x" << depth
            << " checksum=" << checksum << "\n";

  writeVolumeSlices<FloatImage>(output, outputPath, inputNames);
}

static void runFftKernel(const Options& options) {
  if (options.pixelType != "Float32") {
    throw std::runtime_error("fftKernel currently expects --pixel-type Float32");
  }
  if (options.iterations == 0) {
    throw std::runtime_error("--iterations must be positive");
  }

  using FloatImage = itk::Image<float, 3>;
  using FFT = itk::ForwardFFTImageFilter<FloatImage>;
  auto input = makeKernelVolume<float>(options.shape);
  input->Update();
  const auto parsed = parseShape(options.shape);
  const auto width = static_cast<int>(parsed[0]);
  const auto height = static_cast<int>(parsed[1]);
  const auto depth = static_cast<int>(parsed[2]);
  const auto total = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(depth);
  const float* source = input->GetBufferPointer();

  typename FFT::OutputImageType::Pointer itkOutput;
  auto itkStart = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < options.iterations; ++i) {
    auto fft = FFT::New();
    fft->SetInput(input);
    fft->Update();
    itkOutput = fft->GetOutput();
    itkOutput->DisconnectPipeline();
  }
  const std::chrono::duration<double> itkElapsed = std::chrono::steady_clock::now() - itkStart;
  const auto itkChecksum = checksumComplexImage<typename FFT::OutputImageType>(itkOutput);

  std::vector<float> values(total * 2u);
  std::chrono::duration<double> lowlevelElapsed{};
  for (unsigned i = 0; i < options.iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fillInterleavedComplex64FromReal(source, values);
    for (int z = 0; z < depth; ++z) {
      const auto offset = static_cast<std::size_t>(z) * static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 2u;
      const int status = fftwfComplexXYInplace(values.data() + offset, width, height);
      if (status != 0) {
        throw std::runtime_error("lowlevel fftxy failed");
      }
    }
    const int status = fftwfComplexZInplace(values.data(), width, height, depth);
    if (status != 0) {
      throw std::runtime_error("lowlevel fftz failed");
    }
    lowlevelElapsed += std::chrono::steady_clock::now() - start;
  }
  const auto lowlevelChecksum = checksumInterleavedComplex64(values);

  std::cout.precision(9);
  std::cout << std::fixed
            << "variant=cpp-itk shape=" << options.shape
            << " iterations=" << options.iterations
            << " totalSeconds=" << itkElapsed.count()
            << " perIterationSeconds=" << (itkElapsed.count() / static_cast<double>(options.iterations))
            << " checksum=" << itkChecksum << "\n"
            << "variant=lowlevel-xy-z shape=" << options.shape
            << " iterations=" << options.iterations
            << " totalSeconds=" << lowlevelElapsed.count()
            << " perIterationSeconds=" << (lowlevelElapsed.count() / static_cast<double>(options.iterations))
            << " checksum=" << lowlevelChecksum << "\n";
  writeInternalSeconds(itkElapsed + lowlevelElapsed);
}

template <typename T>
static void runTyped(const Options& options) {
  using Image = itk::Image<T, 3>;
  using Mask = itk::Image<std::uint8_t, 3>;
  using Label = itk::Image<std::uint64_t, 3>;

  if (options.operation == "fftKernel") {
    runFftKernel(options);
    return;
  }

  if (options.operation == "fftRoundtripKernel" || options.operation == "fft-roundtrip-kernel") {
    runFftRoundtripKernel(options);
    return;
  }

  if (options.operation == "thresholdKernel") {
    using Threshold = itk::BinaryThresholdImageFilter<Image, Mask>;
    auto input = makeKernelVolume<T>(options.shape);
    auto filter = Threshold::New();
    filter->SetInput(input);
    filter->SetLowerThreshold(static_cast<T>(options.threshold));
    filter->SetUpperThreshold(itk::NumericTraits<T>::max());
    filter->SetInsideValue(1);
    filter->SetOutsideValue(0);
    const auto start = std::chrono::steady_clock::now();
    filter->Update();
    writeInternalSeconds(std::chrono::steady_clock::now() - start);
    return;
  }

  if (options.operation == "thresholdKernelInType") {
    using Threshold = itk::BinaryThresholdImageFilter<Image, Image>;
    auto input = makeKernelVolume<T>(options.shape);
    auto filter = Threshold::New();
    filter->SetInput(input);
    filter->SetLowerThreshold(static_cast<T>(options.threshold));
    filter->SetUpperThreshold(itk::NumericTraits<T>::max());
    filter->SetInsideValue(static_cast<T>(1));
    filter->SetOutsideValue(static_cast<T>(0));
    const auto start = std::chrono::steady_clock::now();
    filter->Update();
    writeInternalSeconds(std::chrono::steady_clock::now() - start);
    return;
  }

  auto paths = tiffFiles(options.input);
  auto input = readVolume<T>(paths);

  if (options.operation == "fft-zarr") {
    using FloatImage = itk::Image<float, 3>;
    using CastToFloat = itk::CastImageFilter<Image, FloatImage>;
    using FFT = itk::ForwardFFTImageFilter<FloatImage>;

    auto cast = CastToFloat::New();
    cast->SetInput(input);

    auto fft = FFT::New();
    fft->SetInput(cast->GetOutput());
    fft->Update();

    using ComplexImage = typename FFT::OutputImageType;
    writeComplex64Zarr<ComplexImage>(fft->GetOutput(), options.output, options.chunkSize);
    return;
  }

  if (options.operation == "fftRoundtripLowlevel" || options.operation == "fft-roundtrip-lowlevel") {
    if constexpr (std::is_same_v<T, float>) {
      runLowlevelFftRoundtripIo(input, options.output, paths);
    } else {
      throw std::runtime_error("fftRoundtripLowlevel currently expects --pixel-type Float32");
    }
    return;
  }

  if (options.operation == "fftRoundtrip" || options.operation == "fft-roundtrip") {
    using FloatImage = itk::Image<float, 3>;
    using CastToFloat = itk::CastImageFilter<Image, FloatImage>;
    using FFT = itk::ForwardFFTImageFilter<FloatImage>;
    using ComplexImage = typename FFT::OutputImageType;
    using IFFT = itk::InverseFFTImageFilter<ComplexImage, FloatImage>;

    auto cast = CastToFloat::New();
    cast->SetInput(input);

    auto fft = FFT::New();
    fft->SetInput(cast->GetOutput());

    auto ifft = IFFT::New();
    ifft->SetInput(fft->GetOutput());

    if constexpr (std::is_same_v<T, float>) {
      writeVolumeSlices<FloatImage>(ifft->GetOutput(), options.output, paths);
    } else {
      using CastToOutput = itk::CastImageFilter<FloatImage, Image>;
      auto outputCast = CastToOutput::New();
      outputCast->SetInput(ifft->GetOutput());
      writeVolumeSlices<Image>(outputCast->GetOutput(), options.output, paths);
    }
    return;
  }

  const auto radiusValue = static_cast<typename Image::SizeValueType>(options.radius);
  typename Image::SizeType radius;
  radius.Fill(radiusValue);

  using Element = itk::BinaryBallStructuringElement<std::uint8_t, 3>;
  Element::RadiusType elementRadius;
  elementRadius.Fill(radiusValue);
  Element element;
  element.SetRadius(elementRadius);
  element.CreateStructuringElement();

  if (options.operation == "threshold") {
    using Threshold = itk::BinaryThresholdImageFilter<Image, Mask>;
    auto filter = Threshold::New();
    filter->SetInput(input);
    filter->SetLowerThreshold(static_cast<T>(options.threshold));
    filter->SetUpperThreshold(itk::NumericTraits<T>::max());
    filter->SetInsideValue(1);
    filter->SetOutsideValue(0);
    writeVolumeSlices<Mask>(filter->GetOutput(), options.output, paths);
    return;
  }

  if (options.operation == "dilate") {
    using Threshold = itk::BinaryThresholdImageFilter<Image, Mask>;
    using Dilate = itk::BinaryDilateImageFilter<Mask, Mask, Element>;
    auto mask = Threshold::New();
    mask->SetInput(input);
    mask->SetLowerThreshold(binaryMaskLowerThreshold<T>());
    mask->SetUpperThreshold(itk::NumericTraits<T>::max());
    mask->SetInsideValue(1);
    mask->SetOutsideValue(0);
    auto filter = Dilate::New();
    filter->SetInput(mask->GetOutput());
    filter->SetKernel(element);
    filter->SetForegroundValue(1);
    writeVolumeSlices<Mask>(filter->GetOutput(), options.output, paths);
    return;
  }

  if (options.operation == "connectedComponents") {
    using Threshold = itk::BinaryThresholdImageFilter<Image, Mask>;
    using Components = itk::ConnectedComponentImageFilter<Mask, Label>;
    using Cast = itk::CastImageFilter<Label, Mask>;
    auto mask = Threshold::New();
    mask->SetInput(input);
    mask->SetLowerThreshold(binaryMaskLowerThreshold<T>());
    mask->SetUpperThreshold(itk::NumericTraits<T>::max());
    mask->SetInsideValue(1);
    mask->SetOutsideValue(0);
    auto components = Components::New();
    components->SetInput(mask->GetOutput());
    auto cast = Cast::New();
    cast->SetInput(components->GetOutput());
    writeVolumeSlices<Mask>(cast->GetOutput(), options.output, paths);
    return;
  }

  typename Image::Pointer output;
  if (options.operation == "copy") {
    output = input;
  } else if (options.operation == "convolve") {
    using Kernel = itk::Image<double, 3>;
    using Filter = itk::ConvolutionImageFilter<Image, Kernel, Image>;
    itk::ConstantBoundaryCondition<Image> boundary;
    boundary.SetConstant(T{});
    auto kernel = makeUniformKernel(options.kernelSize);
    auto filter = Filter::New();
    filter->SetInput(input);
    filter->SetKernelImage(kernel);
    filter->SetBoundaryCondition(&boundary);
    filter->SetOutputRegionMode(Filter::OutputRegionModeEnum::SAME);
    filter->NormalizeOff();
    filter->Update();
    output = filter->GetOutput();
  } else if (options.operation == "median") {
    using Filter = itk::MedianImageFilter<Image, Image>;
    auto filter = Filter::New();
    filter->SetInput(input);
    filter->SetRadius(radius);
    filter->Update();
    output = filter->GetOutput();
  } else {
    throw std::runtime_error("unsupported operation: " + options.operation);
  }

  writeVolumeSlices<Image>(output, options.output, paths);
}

static void writeInternalSeconds(std::chrono::duration<double> elapsed) {
  const char* path = std::getenv("BENCHMARK_INTERNAL_SECONDS_PATH");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  std::ofstream out(path);
  out.precision(9);
  out << std::fixed << elapsed.count();
}

int main(int argc, char** argv) {
  try {
    auto options = parseOptions(argc, argv);
    prepareOutput(options);
    const auto start = std::chrono::steady_clock::now();
    if (options.pixelType == "UInt8") {
      runTyped<std::uint8_t>(options);
    } else if (options.pixelType == "UInt16") {
      runTyped<std::uint16_t>(options);
    } else if (options.pixelType == "Float32") {
      runTyped<float>(options);
    } else {
      throw std::runtime_error("unsupported pixel type: " + options.pixelType);
    }
    if (!isKernelOnlyOperation(options.operation)) {
      writeInternalSeconds(std::chrono::steady_clock::now() - start);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 2;
  }
}
