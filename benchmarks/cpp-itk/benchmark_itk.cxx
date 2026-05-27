#include <algorithm>
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
#include "itkExtractImageFilter.h"
#include "itkFlatStructuringElement.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionConstIterator.h"
#include "itkImageRegionIterator.h"
#include "itkMedianImageFilter.h"
#include "itkMeanImageFilter.h"

namespace fs = std::filesystem;

struct Options {
  std::string operation;
  std::string pixelType;
  fs::path input;
  fs::path output;
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
  options.radius = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--radius", "1")));
  options.threshold = std::stod(argValue(argc, argv, "--threshold", "128"));
  options.kernelSize = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--kernel-size", "3")));
  options.window = static_cast<unsigned>(std::stoul(argValue(argc, argv, "--window", "16")));
  if (options.operation.empty() || options.pixelType.empty() || options.input.empty() || options.output.empty()) {
    throw std::runtime_error("required arguments: --operation --pixel-type --input --output");
  }
  return options;
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

  auto firstReader = Reader::New();
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

  for (std::size_t z = 0; z < paths.size(); ++z) {
    typename Image2::Pointer slice;
    if (z == 0) {
      slice = first;
    } else {
      auto reader = Reader::New();
      reader->SetFileName(paths[z].string());
      reader->Update();
      slice = reader->GetOutput();
    }

    if (slice->GetLargestPossibleRegion().GetSize() != sliceSize) {
      throw std::runtime_error("all TIFF slices must have the same size");
    }

    itk::ImageRegionConstIterator<Image2> src(slice, slice->GetLargestPossibleRegion());
    for (src.GoToBegin(); !src.IsAtEnd(); ++src) {
      const auto index2 = src.GetIndex();
      typename Image3::IndexType index3;
      index3[0] = index2[0];
      index3[1] = index2[1];
      index3[2] = static_cast<typename Image3::IndexValueType>(z);
      volume->SetPixel(index3, src.Get());
    }
  }

  return volume;
}

template <typename Image3>
static void writeVolumeSlices(typename Image3::Pointer volume, const fs::path& output, const std::vector<fs::path>& inputNames) {
  using Image2 = itk::Image<typename Image3::PixelType, 2>;
  using Extract = itk::ExtractImageFilter<Image3, Image2>;
  using Writer = itk::ImageFileWriter<Image2>;

  volume->Update();
  const auto region = volume->GetLargestPossibleRegion();
  auto size = region.GetSize();
  auto index = region.GetIndex();
  const auto depth = size[2];
  size[2] = 0;

  for (typename Image3::SizeValueType z = 0; z < depth; ++z) {
    index[2] = static_cast<typename Image3::IndexValueType>(z);
    typename Image3::RegionType sliceRegion;
    sliceRegion.SetIndex(index);
    sliceRegion.SetSize(size);

    auto extract = Extract::New();
    extract->SetInput(volume);
    extract->SetExtractionRegion(sliceRegion);
    extract->SetDirectionCollapseToSubmatrix();

    auto writer = Writer::New();
    writer->SetFileName((output / inputNames[static_cast<std::size_t>(z)].filename()).string());
    writer->SetInput(extract->GetOutput());
    writer->UseCompressionOff();
    writer->Update();
  }
}

template <typename T>
static void runTyped(const Options& options) {
  using Image = itk::Image<T, 3>;
  using Mask = itk::Image<std::uint8_t, 3>;
  using Label = itk::Image<std::uint64_t, 3>;

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

  using Element = itk::FlatStructuringElement<3>;
  Element::RadiusType elementRadius;
  elementRadius.Fill(radiusValue);
  auto element = Element::Ball(elementRadius);

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
  } else if (options.operation == "uniformConvolve") {
    using Filter = itk::MeanImageFilter<Image, Image>;
    auto filter = Filter::New();
    const auto meanRadiusValue = static_cast<typename Image::SizeValueType>(std::max(1u, options.kernelSize) / 2u);
    typename Image::SizeType meanRadius;
    meanRadius.Fill(meanRadiusValue);
    filter->SetInput(input);
    filter->SetRadius(meanRadius);
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
    writeInternalSeconds(std::chrono::steady_clock::now() - start);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << std::endl;
    return 2;
  }
}
