// To run, remember to:
// export DYLD_LIBRARY_PATH=./StackPipeline/lib:$(pwd)/bin/Debug/net8.0
open StackProcessing

[<EntryPoint>]
let main arg =
    let availableMemory = 2UL * 1024UL * 1024UL *1024UL // 1MB for example

    let src = 
        if arg.Length > 0 && arg[0] = "debug" then
            debug availableMemory
        else
            source availableMemory
    let width, height, depth, input, output = 
        if arg.Length > 1 then
            let n = (int arg[1]) / 3 |> pown 2 |> uint 
            n, n, n, $"image{arg[1]}", $"result{arg[1]}"
        else
            64u, 64u, 64u, "image18", "result18"
    let tmp = "tmp"

    let wsz = (depth/4u)

    src
    |> read<uint8> ("../"+input) ".tiff"
    >=> imageDivScalar 255uy
    >=> connectedComponents wsz
    >=> cast<uint64,uint8>
    // Tiff supporst uint8, int8, uint16, int16, and float32
    >=> write ("../"+tmp) ".tiff"
    |> sink

    let transTbl =
        src
        |> getConnectedChunkNeighbours ("../"+tmp) ".tiff" wsz
        >=> readFilePairs<uint64> true 
        >=> makeAdjacencyGraph ()
        >=> tapIt (fun (i,g) -> sprintf "Vertices\n%A\nEdges\n%A\n" (g|>simpleGraph.vertices|>Set.toList|>List.sort) (g|>simpleGraph.edges|>Set.toList|>List.sort))
        >=> makeTranslationTable ()
        |> drain
    printfn "Translation Table drain:\n%A" transTbl
    
    src
    |> getFilenames ("../"+tmp) ".tiff" Array.sort
    >=> updateConnectedComponents wsz transTbl
    >=> write ("../"+output) ".tiff"
    |> sink

    0
