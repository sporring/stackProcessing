module RunSamplesCommon

open System
open System.IO
open System.Text.RegularExpressions

let repositoryRootFromSamplesRoot samplesRoot =
    let cwd = Directory.GetCurrentDirectory()

    if File.Exists(Path.Combine(cwd, "StackProcessing.sln")) then
        cwd
    elif String.Equals(Path.GetFileName(Path.GetFullPath samplesRoot), "samples", StringComparison.OrdinalIgnoreCase) then
        Directory.GetParent(Path.GetFullPath samplesRoot).FullName
    else
        cwd

let runOutputRoot samplesRoot =
    Path.Combine(repositoryRootFromSamplesRoot samplesRoot, "tmp")

let sampleTempRoot samplesRoot =
    Path.Combine(Path.GetFullPath samplesRoot, "tmp")

let relativePath root path =
    Path.GetRelativePath(root, path).Replace(Path.DirectorySeparatorChar, '/')

let safeName (value: string) =
    Regex.Replace(value.Replace('\\', '/'), @"[^A-Za-z0-9_.-]+", "_")
