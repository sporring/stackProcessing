% Compare StackProcessing binary erosion with MATLAB's 3D morphology.
%
% Run from the repository root or directly from this folder:
%   run("samples/erode/erode_matlab_compare.m")
%
% The input is interpreted as a binary UInt8 stack where non-zero pixels are
% foreground. The result is written as UInt8 TIFF slices with foreground 255
% for easy visual inspection.

tic
radius = 16;

scriptDir = fileparts(mfilename("fullpath"));
inputDir = fullfile(scriptDir, "..", "data", "rotatingBoxes");
outputDir = fullfile(scriptDir, "..", "data", "rotatingBoxes_matlab_erode_r16");

files = dir(fullfile(inputDir, "*.tif*"));
if isempty(files)
    error("No TIFF slices found in %s", inputDir);
end

[~, order] = sort({files.name});
files = files(order);

firstSlice = imread(fullfile(inputDir, files(1).name));
if ndims(firstSlice) > 2
    firstSlice = rgb2gray(firstSlice);
end

volume = false([size(firstSlice, 1), size(firstSlice, 2), numel(files)]);
for z = 1:numel(files)
    slice = imread(fullfile(inputDir, files(z).name));
    if ndims(slice) > 2
        slice = rgb2gray(slice);
    end
    volume(:, :, z) = slice > 0;
end

fprintf("Read %d slices of size %dx%d from %s\n", numel(files), size(volume, 2), size(volume, 1), inputDir);

se = strel("sphere", radius);
eroded = imerode(volume, se);

if ~exist(outputDir, "dir")
    mkdir(outputDir);
end

for z = 1:size(eroded, 3)
    outputSlice = uint8(eroded(:, :, z)) * uint8(255);
    imwrite(outputSlice, fullfile(outputDir, files(z).name), "Compression", "none");
end

fprintf("Wrote MATLAB sphere-radius-%d erosion to %s (%d secs)\n", radius, outputDir, toc);
