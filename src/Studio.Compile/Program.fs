namespace Studio.Compile

open System
open System.IO
open Studio.Compiler
open Studio.Graph

module Program =

    let private usage =
        String.concat
            Environment.NewLine
            [ "Usage:"
              "  dotnet run --project src/Studio.Compile -- <input.json> <output.fs>"
              ""
              "Compiles a Studio pipeline graph JSON file to StackProcessing DSL/F# text." ]

    let private ensureParentDirectory (path: string) =
        let fullPath = Path.GetFullPath(path)
        let directory = Path.GetDirectoryName(fullPath)

        if not (String.IsNullOrWhiteSpace directory) then
            Directory.CreateDirectory(directory) |> ignore

    [<EntryPoint>]
    let main argv =
        try
            match argv with
            | [| inputPath; outputPath |] ->
                if not (File.Exists inputPath) then
                    eprintfn "Input graph JSON file does not exist: %s" inputPath
                    2
                else
                    let graph = PipelineGraphStorage.load inputPath
                    let dsl = PipelineCodeGenerator.generateSavedGraph graph
                    ensureParentDirectory outputPath
                    File.WriteAllText(outputPath, dsl)
                    0
            | [| "--help" |]
            | [| "-h" |]
            | [| "/?" |] ->
                printfn "%s" usage
                0
            | _ ->
                eprintfn "%s" usage
                1
        with ex ->
            eprintfn "Could not compile Studio graph: %s" ex.Message
            1
