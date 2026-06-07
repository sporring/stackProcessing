function bench_stack(varargin)
%BENCH_STACK MATLAB TIFF-stack benchmark backend.
% Example:
%   bench_stack("operation","threshold","pixelType","UInt8","input","in","output","out")

args = struct("operation", "", "pixelType", "", "input", "", "output", "", "radius", 1, "kernelSize", 3, "threshold", 128, "window", 16, "kernelMode", "uniform");
for k = 1:2:numel(varargin)
    key = char(varargin{k});
    value = varargin{k + 1};
    args.(key) = value;
end

if args.operation == "" || args.input == "" || args.output == ""
    error("operation, input, and output are required");
end

internalTimer = tic;

if ~exist(args.output, "dir")
    mkdir(args.output);
end
delete(fullfile(args.output, "*.tif*"));

files = dir(fullfile(args.input, "*.tif*"));
[~, order] = sort({files.name});
files = files(order);
if isempty(files)
    error("no TIFF files found in %s", args.input);
end

radius = numericArg(args.radius);
kernelSize = max(1, 2 * radius + 1);

first = imread(fullfile(files(1).folder, files(1).name));
volume = zeros([size(first), numel(files)], "like", first);
volume(:, :, 1) = first;
for i = 2:numel(files)
    volume(:, :, i) = imread(fullfile(files(i).folder, files(i).name));
end

switch char(args.operation)
    case "copy"
        out = volume;
    case "threshold"
        out = uint8(volume >= numericArg(args.threshold));
    case "convolve"
        kernelSize = max(1, numericArg(args.kernelSize));
        if string(args.kernelMode) == "random"
            rng(1, "twister");
            kernel = randn(kernelSize, kernelSize, kernelSize, "double");
        else
            kernel = ones(kernelSize, kernelSize, kernelSize, "double") ./ (kernelSize ^ 3);
        end
        disp("convn kernel size: " + mat2str(size(kernel)) + ", mode: " + string(args.kernelMode));
        computeVolume = double(volume);
        out = cast(convn(computeVolume, kernel, "same"), class(volume));
    case "median"
        out = medfilt3(volume, [kernelSize kernelSize kernelSize], "symmetric");
    case "dilate"
        se = strel("sphere", radius);
        out = uint8(imdilate(volume >= 128, se));
    case "connectedComponents"
        components = bwconncomp(volume >= 128, 6);
        out = uint8(mod(labelmatrix(components), 256));
    otherwise
        error("unsupported operation %s", args.operation);
end

for i = 1:size(out, 3)
    writeTiffSlice(out(:, :, i), fullfile(args.output, files(i).name));
end
writeInternalSeconds(toc(internalTimer));
end

function writeInternalSeconds(seconds)
path = getenv("BENCHMARK_INTERNAL_SECONDS_PATH");
if ~isempty(path)
    fid = fopen(path, "w");
    if fid < 0
        error("could not open internal timing path %s", path);
    end
    cleanup = onCleanup(@() fclose(fid));
    fprintf(fid, "%.9f", seconds);
end
end

function value = numericArg(value)
if ischar(value) || isstring(value)
    value = str2double(value);
else
    value = double(value);
end
if ~isscalar(value) || isnan(value)
    error("expected scalar numeric argument");
end
end

function writeTiffSlice(slice, path)
t = Tiff(path, "w");
cleanup = onCleanup(@() close(t));
tags.ImageLength = size(slice, 1);
tags.ImageWidth = size(slice, 2);
tags.Photometric = Tiff.Photometric.MinIsBlack;
tags.SamplesPerPixel = 1;
tags.BitsPerSample = bitsPerSample(slice);
tags.SampleFormat = sampleFormat(slice);
tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
tags.Compression = Tiff.Compression.None;
tags.RowsPerStrip = size(slice, 1);
setTag(t, tags);
write(t, slice);
end

function bits = bitsPerSample(slice)
switch class(slice)
    case {"uint8", "int8"}
        bits = 8;
    case {"uint16", "int16"}
        bits = 16;
    case {"uint32", "int32", "single"}
        bits = 32;
    case {"uint64", "int64", "double"}
        bits = 64;
    otherwise
        error("unsupported TIFF slice class %s", class(slice));
end
end

function format = sampleFormat(slice)
switch class(slice)
    case {"uint8", "uint16", "uint32", "uint64"}
        format = Tiff.SampleFormat.UInt;
    case {"int8", "int16", "int32", "int64"}
        format = Tiff.SampleFormat.Int;
    case {"single", "double"}
        format = Tiff.SampleFormat.IEEEFP;
    otherwise
        error("unsupported TIFF slice class %s", class(slice));
end
end
