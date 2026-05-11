#r "src/Studio.Graph/bin/Debug/net10.0/Studio.Graph.dll"
#r "src/Studio.Compiler/bin/Debug/net10.0/Studio.Compiler.dll"

open System
open System.IO
open Studio.Graph
open Studio.Compiler

let root = Path.Combine(Directory.GetCurrentDirectory(), "samples")
let files = Directory.GetFiles(root, "*.json", SearchOption.AllDirectories) |> Array.sort

let mutable failures = []

for file in files do
    try
        let code = PipelineGraphStorage.load file |> PipelineCodeGenerator.generateSavedGraph
        if code.StartsWith("// Cannot generate F# yet") then
            failures <- (file, code) :: failures
        else
            printfn "OK %s" (Path.GetRelativePath(Directory.GetCurrentDirectory(), file))
    with ex ->
        failures <- (file, ex.Message) :: failures

if failures.Length > 0 then
    for file, message in List.rev failures do
        eprintfn "FAILED %s\n%s" (Path.GetRelativePath(Directory.GetCurrentDirectory(), file)) message
    failwithf "%d sample graph(s) failed" failures.Length
