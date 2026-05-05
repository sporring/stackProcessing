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
                    [ p "operation" "-" false
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
            Expect.stringContains code ">>=> subPair" "The selected subtraction operation should lower to subPair."

        testCase "connected component pair stream writes chunk labels through teeFst before reducing" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let connected =
                node "connected" "ConnectedComponents" [ p "windowSize" "15" false ]

            let writeChunks =
                node "writeChunks" "WriteChunkSlices"
                    [ p "output" "tmp" false
                      p "suffix" ".mha" false
                      p "windowSize" "15" false ]

            let table =
                node "table" "ComponentTranslationTable" [ p "windowSize" "15" false ]

            let code =
                graph
                    [ read; connected; writeChunks; table ]
                    [ edge "read" "output" 0 "connected" "input" 0
                      edge "connected" "output" 0 "writeChunks" "input" 0
                      edge "writeChunks" "output" 0 "table" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> connectedComponents 15u" "Connected components should produce label/count pairs."
            Expect.stringContains code ">=> teeFst (writeChunkSlices \"tmp\" \".mha\" 15u)" "Chunk writing should be an explicit tee over the first tuple element."
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
