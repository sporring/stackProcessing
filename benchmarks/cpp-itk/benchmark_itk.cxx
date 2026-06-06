#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryDilateImageFilter.h"
#include "itkCastImageFilter.h"
#include "itkConnectedComponentImageFilter.h"
#include "itkConstantBoundaryCondition.h"
#include "itkConvolutionImageFilter.h"
#include "itkBinaryBallStructuringElement.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
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
  if (options.operation.empty() || options.pixelType.empty()) {
    throw std::runtime_error("required arguments: --operation --pixel-type");
  }
  if (options.operation != "thresholdKernel" && options.operation != "thresholdKernelInType" && (options.input.empty() || options.output.empty())) {
    throw std::runtime_error("required arguments for IO operations: --input --output");
  }
  return options;
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

template <typename T>
static void runTyped(const Options& options) {
  using Image = itk::Image<T, 3>;
  using Mask = itk::Image<std::uint8_t, 3>;
  using Label = itk::Image<std::uint64_t, 3>;

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

  fs::create_directories(options.output);
  for (const auto& entry : fs::directory_iterator(options.output)) {
    if (entry.is_regular_file()) {
      fs::remove(entry.path());
    }
  }

  auto paths = tiffFiles(options.input);
  auto input = readVolume<T>(paths);

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
    if (options.operation != "thresholdKernel" && options.operation != "thresholdKernelInType") {
      writeInternalSeconds(std::chrono::steady_clock::now() - start);
    }
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 2;
  }
}
