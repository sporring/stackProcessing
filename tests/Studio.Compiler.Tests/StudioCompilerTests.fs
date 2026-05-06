module Tests.StudioCompilerTests

open Expecto
open Studio.Graph
open Studio.Compiler

let private p key value useInput =
    { Key = key; Value = value; UseInput = useInput }

let private node id functionId parameters =
    { Id = id
      FunctionId = functionId
      X = 0.0
      Y = 0.0
      Parameters = parameters |> List.toArray }

let private edge fromNode fromKind fromPort toNode toKind toPort =
    { FromNode = fromNode
      FromKind = fromKind
      FromPort = fromPort
      ToNode = toNode
      ToKind = toKind
      ToPort = toPort }

let private graph nodes edges =
    { Version = 1
      Nodes = nodes |> List.toArray
      Edges = edges |> List.toArray }

[<Tests>]
let generatorSuite =
    testList "Studio.Compiler PipelineCodeGenerator" [
        testCase "read write pipeline generates source and sink" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph [ read; write ] [ edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "open StackProcessing" "Generated code should open StackProcessing."
            Expect.stringContains code "source 1073741824UL" "Read node should generate source with uint64 available memory."
            Expect.stringContains code "|> read<uint8> \"input\" \".tiff\"" "Read node should generate typed read."
            Expect.stringContains code ">=> write \"output\" \".tiff\"" "Write node should generate write stage."
            Expect.stringContains code "|> sink" "Terminal write should be sunk."

        testCase "slab read and write boxes lower to slab DSL functions" <| fun _ ->
            let read =
                node "read" "ReadSlab"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "chunks" false
                      p "suffix" ".mha" false ]

            let write =
                node "write" "WriteInSlabs"
                    [ p "output" "chunks-out" false
                      p "suffix" ".mha" false
                      p "chunkX" "12" false
                      p "chunkY" "13" false
                      p "chunkZ" "14" false ]

            let code =
                graph [ read; write ] [ edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> readSlab<uint8> \"chunks\" \".mha\"" "ReadSlab should generate the slab reader."
            Expect.stringContains code ">=> writeInSlabs \"chunks-out\" \".mha\" 12u 13u 14u" "WriteInSlabs should generate the slab writer wrapper."

        testCase "zarr boxes lower to zarr DSL functions" <| fun _ ->
            let info =
                node "info" "GetZarrInfo"
                    [ p "input" "input.zarr" false
                      p "multiscaleIndex" "0" false
                      p "datasetIndex" "1" false ]

            let read =
                node "read" "ReadZarrSlab"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt16" false
                      p "input" "input.zarr" false
                      p "slabDepth" "8" false
                      p "multiscaleIndex" "0" false
                      p "datasetIndex" "1" false
                      p "timepoint" "2" false
                      p "channel" "3" false
                      p "maxParallelChunks" "4" false ]

            let write =
                node "write" "WriteZarr"
                    [ p "output" "output.zarr" false
                      p "name" "" true
                      p "depth" "32" false
                      p "chunkX" "16" false
                      p "chunkY" "17" false
                      p "chunkZ" "8" false
                      p "physicalSizeX" "0.5" false
                      p "physicalSizeY" "0.5" false
                      p "physicalSizeZ" "2.0" false
                      p "maxConcurrentWrites" "2" false ]

            let code =
                graph
                    [ info; read; write ]
                    [ edge "read" "output" 0 "write" "input" 0
                      edge "info" "reducerOutput" 2 "write" "parameterInput" 1 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let ChunkInfo0 = getZarrInfo \"input.zarr\" 0 1" "GetZarrInfo should generate a metadata binding."
            Expect.stringContains code "|> readZarrSlab<uint16> \"input.zarr\" 8u 0 1 2 3 4" "ReadZarrSlab should generate the Zarr slab reader."
            Expect.stringContains code ">=> writeZarr \"output.zarr\" ChunkInfo0.topLeftInfo.componentType 32u 16u 17u 8u 0.5 0.5 2.0 2" "WriteZarr should accept linked Zarr metadata."
            Expect.stringContains code "|> sink" "Terminal WriteZarr should be sunk."

        testCase "linked string scalar is emitted before read and used unquoted" <| fun _ ->
            let scalar =
                node "scalar" "Scalar"
                    [ p "type" "String" false
                      p "value" "input" false ]

            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "" true
                      p "suffix" ".tiff" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ scalar; read; write ]
                    [ edge "scalar" "scalarOutput" 0 "read" "parameterInput" 2
                      edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let String0 = \"input\"" "String scalar should be bound."
            Expect.stringContains code "|> read<uint8> String0 \".tiff\"" "Linked scalar should be used as an identifier, not a string literal."
            Expect.isLessThan (code.IndexOf("let String0")) (code.IndexOf("|> read<uint8>")) "Scalar binding should appear before dependent pipeline."

        testCase "file directory source is emitted as string binding before dependent read" <| fun _ ->
            let fileDirectory =
                node "path" "FileDirectory"
                    [ p "kind" "Directory" false
                      p "value" "../image18" false ]

            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "" true
                      p "suffix" ".tiff" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ fileDirectory; read; write ]
                    [ edge "path" "scalarOutput" 0 "read" "parameterInput" 2
                      edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let String0 = \"../image18\"" "File/directory source should be stored as a string binding."
            Expect.stringContains code "|> read<uint8> String0 \".tiff\"" "Read should consume the selected path binding."
            Expect.isLessThan (code.IndexOf("let String0")) (code.IndexOf("|> read<uint8>")) "The selected path should be bound before dependent read."

        testCase "scalar operation binding follows dependency order" <| fun _ ->
            let scalar =
                node "scalar" "Scalar"
                    [ p "type" "Float64" false
                      p "value" "255" false ]

            let op =
                node "op" "ScalarOp"
                    [ p "operation" "/" false
                      p "type" "Float64" false
                      p "a" "" true
                      p "b" "2" false ]

            let print =
                node "print" "Print"
                    [ p "format" "{input1}" false
                      p "input1" "" true
                      p "input2" "input2" false ]

            let code =
                graph
                    [ scalar; op; print ]
                    [ edge "scalar" "scalarOutput" 0 "op" "parameterInput" 2
                      edge "op" "scalarOutput" 0 "print" "parameterInput" 1 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Float640 = 255.0" "Scalar should get typed literal."
            Expect.stringContains code "let Float641 = (Float640 / 2.0)" "ScalarOp should depend on previous scalar binding."
            Expect.isLessThan (code.IndexOf("let Float640")) (code.IndexOf("let Float641")) "Dependency binding should come first."
            Expect.stringContains code "printfn $\"{Float641}\"" "Print should emit interpolated printfn."

        testCase "scalar function binding lowers selected F# function" <| fun _ ->
            let scalar =
                node "scalar" "Scalar"
                    [ p "type" "Float64" false
                      p "value" "0" false ]

            let fn =
                node "fn" "ScalarFunction"
                    [ p "function" "cos" false
                      p "a" "" true ]

            let square =
                node "square" "ScalarFunction"
                    [ p "function" "square" false
                      p "a" "" true ]

            let print =
                node "print" "Print"
                    [ p "format" "{input1} {input2}" false
                      p "input1" "" true
                      p "input2" "" true ]

            let code =
                graph
                    [ scalar; fn; square; print ]
                    [ edge "scalar" "scalarOutput" 0 "fn" "parameterInput" 1
                      edge "fn" "scalarOutput" 0 "square" "parameterInput" 1
                      edge "fn" "scalarOutput" 0 "print" "parameterInput" 1
                      edge "square" "scalarOutput" 0 "print" "parameterInput" 2 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Float641 = (cos Float640)" "ScalarFunction should lower F# function selection."
            Expect.stringContains code "let Float642 = (Float641 * Float641)" "ScalarFunction should lower square as multiplication."
            Expect.stringContains code "printfn $\"{Float641} {Float642}\"" "Print should consume scalar function outputs."

        testCase "standard numeric names lower to Math constants but strings stay literal" <| fun _ ->
            let numeric =
                node "numeric" "Scalar"
                    [ p "type" "Float64" false
                      p "value" "pi" false ]

            let text =
                node "text" "Scalar"
                    [ p "type" "String" false
                      p "value" "e" false ]

            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let threshold =
                node "threshold" "Threshold"
                    [ p "type" "Float64" false
                      p "lower" "e" false
                      p "upper" "pi" false ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let print =
                node "print" "Print"
                    [ p "format" "{input1} {input2}" false
                      p "input1" "" true
                      p "input2" "" true ]

            let code =
                graph
                    [ numeric; text; read; threshold; write; print ]
                    [ edge "numeric" "scalarOutput" 0 "print" "parameterInput" 1
                      edge "text" "scalarOutput" 0 "print" "parameterInput" 2
                      edge "read" "output" 0 "threshold" "input" 0
                      edge "threshold" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Float640 = System.Math.PI" "Numeric scalar pi should lower to Math.PI."
            Expect.stringContains code "let String0 = \"e\"" "String scalar e should remain a string literal."
            Expect.stringContains code ">=> threshold System.Math.E System.Math.PI" "Numeric parameters should lower standard names to Math constants."

        testCase "getStackInfo binds file info fields for scalar outputs" <| fun _ ->
            let path =
                node "path" "Scalar"
                    [ p "type" "String" false
                      p "value" "input" false ]

            let info =
                node "info" "GetStackInfo"
                    [ p "input" "" true
                      p "suffix" ".tiff" false ]

            let print =
                node "print" "Print"
                    [ p "format" "{input1} {input2} {input3}" false
                      p "input1" "" true
                      p "input2" "" true
                      p "input3" "" true ]

            let code =
                graph
                    [ path; info; print ]
                    [ edge "path" "scalarOutput" 0 "info" "parameterInput" 0
                      edge "info" "reducerOutput" 4 "print" "parameterInput" 1
                      edge "info" "reducerOutput" 2 "print" "parameterInput" 2
                      edge "info" "reducerOutput" 0 "print" "parameterInput" 3 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let String0 = \"input\"" "Linked stack-info input should keep its string binding."
            Expect.stringContains code "let StackInfo0 = getStackInfo String0 \".tiff\"" "getStackInfo should bind FileInfo once."
            Expect.stringContains code "printfn $\"{StackInfo0.size[0]} {StackInfo0.componentType} {StackInfo0.dimensions}\"" "FileInfo output pins should map to field expressions."

        testCase "getChunkInfo binds chunk layout fields for scalar outputs" <| fun _ ->
            let info =
                node "info" "GetChunkInfo"
                    [ p "input" "chunks" false
                      p "suffix" ".mha" false ]

            let print =
                node "print" "Print"
                    [ p "format" "{input1} {input2} {input3}" false
                      p "input1" "" true
                      p "input2" "" true
                      p "input3" "" true ]

            let code =
                graph
                    [ info; print ]
                    [ edge "info" "reducerOutput" 4 "print" "parameterInput" 1
                      edge "info" "reducerOutput" 7 "print" "parameterInput" 2
                      edge "info" "reducerOutput" 2 "print" "parameterInput" 3 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let ChunkInfo0 = getChunkInfo \"chunks\" \".mha\"" "getChunkInfo should bind ChunkInfo once."
            Expect.stringContains code "printfn $\"{ChunkInfo0.chunks[0]} {ChunkInfo0.size[0]} {ChunkInfo0.topLeftInfo.componentType}\"" "ChunkInfo output pins should map to field expressions."

        testCase "new image processing boxes lower to StackProcessing stages" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let fill =
                node "fill" "BinaryFillHoles" [ p "windowSize" "5" false ]
            let relabel =
                node "relabel" "RelabelComponents"
                    [ p "minimumObjectSize" "10" false
                      p "windowSize" "5" false ]
            let watershed =
                node "watershed" "Watershed"
                    [ p "level" "1.25" false
                      p "windowSize" "7" false ]
            let signed =
                node "signed" "SignedDistanceMap" [ p "windowSize" "9" false ]
            let otsu =
                node "otsu" "OtsuThreshold" [ p "windowSize" "11" false ]
            let moments =
                node "moments" "MomentsThreshold" [ p "windowSize" "13" false ]
            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; fill; relabel; watershed; signed; otsu; moments; write ]
                    [ edge "read" "output" 0 "fill" "input" 0
                      edge "fill" "output" 0 "relabel" "input" 0
                      edge "relabel" "output" 0 "watershed" "input" 0
                      edge "watershed" "output" 0 "signed" "input" 0
                      edge "signed" "output" 0 "otsu" "input" 0
                      edge "otsu" "output" 0 "moments" "input" 0
                      edge "moments" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> binaryFillHoles 5u" "binaryFillHoles should lower with its window size."
            Expect.stringContains code ">=> relabelComponents 10u 5u" "relabelComponents should lower with size threshold and window size."
            Expect.stringContains code ">=> watershed 1.25 7u" "watershed should lower with level and window size."
            Expect.stringContains code ">=> signedDistanceMap 9u" "signedDistanceMap should lower with its window size."
            Expect.stringContains code ">=> otsuThreshold 11u" "otsuThreshold should lower with its window size."
            Expect.stringContains code ">=> momentsThreshold 13u" "momentsThreshold should lower with its window size."

        testCase "convolve lowers to StackProcessing stage" <| fun _ ->
            let readImage =
                node "image" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "image" false
                      p "suffix" ".tiff" false ]

            let convolve =
                node "convolve" "Convolve"
                    [ p "kernel" "kernelImage" false
                      p "outputRegionMode" "Same" false
                      p "boundaryCondition" "ZeroFluxNeumannPad" false
                      p "windowSize" "8" false ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ readImage; convolve; write ]
                    [ edge "image" "output" 0 "convolve" "input" 0
                      edge "convolve" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> convolve kernelImage (Some ImageFunctions.Same) (Some ImageFunctions.ZeroFluxNeumannPad) (Some 8u)" "Convolve should call the StackProcessing convolve stage."

        testCase "print format unescapes newline and maps linked names" <| fun _ ->
            let stats =
                node "stats" "ComputeStats" []

            let print =
                node "print" "Print"
                    [ p "format" "Pixels: {NumPixels}\\nMean: {Mean}" false
                      p "input1" "" true
                      p "input2" "" true ]

            let code =
                graph
                    [ stats; print ]
                    [ edge "stats" "reducerOutput" 0 "print" "parameterInput" 1
                      edge "stats" "reducerOutput" 1 "print" "parameterInput" 2 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let ImageStats0 =" "Linked reducer output should create a stats binding."
            Expect.stringContains code "{ImageStats0.NumPixels}" "Format placeholder should map to NumPixels expression."
            Expect.stringContains code "{ImageStats0.Mean}" "Format placeholder should map to Mean expression."
            Expect.stringContains code $"Pixels: {{ImageStats0.NumPixels}}{System.Environment.NewLine}Mean: {{ImageStats0.Mean}}" "Generated F# string should contain a literal newline."

        testCase "image op image lowers selected operation to pair stage" <| fun _ ->
            let assertOperation operation expectedPairFunction =
                let readA =
                    node "readA" "Read"
                        [ p "availableMemory" "1024" false
                          p "type" "UInt8" false
                          p "input" "a" false
                          p "suffix" ".tiff" false ]

                let readB =
                    node "readB" "Read"
                        [ p "availableMemory" "1024" false
                          p "type" "UInt8" false
                          p "input" "b" false
                          p "suffix" ".tiff" false ]

                let op =
                    node "op" "ImageOpImage"
                        [ p "operation" operation false
                          p "type" "UInt8" false ]

                let write =
                    node "write" "Write"
                        [ p "output" "out" false
                          p "suffix" ".tiff" false ]

                let code =
                    graph
                        [ readA; readB; op; write ]
                        [ edge "readA" "output" 0 "op" "input" 0
                          edge "readB" "output" 0 "op" "input" 1
                          edge "op" "output" 0 "write" "input" 0 ]
                    |> PipelineCodeGenerator.generateSavedGraph

                Expect.stringContains code "||> zip" "Independent image inputs should be zipped."
                Expect.stringContains code $">>=> {expectedPairFunction}" $"The selected {operation} operation should lower to {expectedPairFunction}."

            assertOperation "-" "subPair"
            assertOperation "max" "maxOfPair"
            assertOperation "min" "minOfPair"

        testCase "unary image function lowers selected function to stage" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let unary =
                node "unary" "UnaryImageFunction"
                    [ p "function" "cos" false ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; unary; write ]
                    [ edge "read" "output" 0 "unary" "input" 0
                      edge "unary" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> cos" "The selected unary image function should lower to the matching StackProcessing stage."

        testCase "connected component pair stream writes chunk labels through teeFst before reducing" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let connected =
                node "connected" "ConnectedComponents" [ p "windowSize" "15" false ]

            let writeSlabSlices =
                node "writeSlabSlices" "WriteSlabSlices"
                    [ p "output" "tmp" false
                      p "suffix" ".mha" false
                      p "windowSize" "15" false ]

            let table =
                node "table" "ComponentTranslationTable" [ p "windowSize" "15" false ]

            let code =
                graph
                    [ read; connected; writeSlabSlices; table ]
                    [ edge "read" "output" 0 "connected" "input" 0
                      edge "connected" "output" 0 "writeSlabSlices" "input" 0
                      edge "writeSlabSlices" "output" 0 "table" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> connectedComponents 15u" "Connected components should produce label/count pairs."
            Expect.stringContains code ">=> teeFst (writeSlabSlices \"tmp\" \".mha\" 15u)" "Slab-slice writing should be an explicit tee over the first tuple element."
            Expect.stringContains code ">=> makeConnectedComponentTranslationTable 15u" "Translation table should consume the pair stream."
            Expect.stringContains code "|> drain" "The reducer should be drained."

        testCase "tap connected to print becomes tapIt lambda with stream value name" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let tap =
                node "tap" "Tap" [ p "label" "debug" false ]

            let print =
                node "print" "Print"
                    [ p "format" "Image: {I}" false
                      p "input1" "" true ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; tap; print; write ]
                    [ edge "read" "output" 0 "tap" "input" 0
                      edge "tap" "scalarOutput" 0 "print" "parameterInput" 1
                      edge "tap" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> tapIt (fun I -> $\"Image: {I}\")" "Tap should absorb the print format as a lambda."
            Expect.isFalse (code.Contains("printfn $\"Image:")) "The helper print node should not also be emitted as a terminal program."

        testCase "chart over histogram data emits visualization helper and bound histogram" <| fun _ ->
            let histogram =
                node "histogram" "HistogramData" []

            let chart =
                node "chart" "Chart"
                    [ p "kind" "Line" false
                      p "input" "" true ]

            let code =
                graph
                    [ histogram; chart ]
                    [ edge "histogram" "reducerOutput" 0 "chart" "parameterInput" 1 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "open Plotly.NET" "Charts need Plotly.NET."
            Expect.stringContains code "let showChart kind points =" "Chart helper should be emitted once."
            Expect.stringContains code "let Histogram0 =" "Linked histogram data should be bound before use."
            Expect.stringContains code "showChart \"Line\" Histogram0" "Chart should use the selected chart kind and linked histogram value."
    ]
