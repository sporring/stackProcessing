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
            Expect.stringContains code "debug 1u 1073741824UL" "Read node should generate a debug source with uint64 available memory."
            Expect.stringContains code "|> read<uint8> \"input\" \".tiff\"" "Read node should generate typed read."
            Expect.stringContains code ">=> write \"output\" \".tiff\"" "Write node should generate write stage."
            Expect.stringContains code "|> sink" "Terminal write should be sunk."

        testCase "readVolume writeVolume pipeline generates streaming volume conversion" <| fun _ ->
            let read =
                node "readVolume" "ReadVolume"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt16" false
                      p "input" "input.tiff" false ]

            let write =
                node "writeVolume" "WriteVolume"
                    [ p "output" "output.tiff" false ]

            let code =
                graph [ read; write ] [ edge "readVolume" "output" 0 "writeVolume" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> readVolume<uint16> \"input.tiff\"" "ReadVolume should generate a typed volume source."
            Expect.stringContains code ">=> writeVolume \"output.tiff\"" "WriteVolume should generate a streaming volume writer."
            Expect.stringContains code "|> sink" "Terminal WriteVolume should be sunk."

        testCase "normalNoise source lowers to synthetic noise source" <| fun _ ->
            let noise =
                node "noise" "NormalNoise"
                    [ p "availableMemory" "2048" false
                      p "type" "Float32" false
                      p "width" "12" false
                      p "height" "13" false
                      p "depth" "4" false
                      p "mean" "5.0" false
                      p "std" "0.25" false ]

            let write =
                node "write" "Write"
                    [ p "output" "noise" false
                      p "suffix" ".tiff" false ]

            let code =
                graph [ noise; write ] [ edge "noise" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "debug 1u 2048UL" "NormalNoise should generate a debug source with available memory."
            Expect.stringContains code "|> normalNoise<float32> 12u 13u 4u 5.0 0.25" "NormalNoise should generate typed dimensions and distribution parameters."

            let saltCode =
                graph
                    [ node "salt" "SaltAndPepperNoise"
                        [ p "availableMemory" "2048" false
                          p "type" "Float32" false
                          p "width" "12" false
                          p "height" "13" false
                          p "depth" "4" false
                          p "probability" "0.01" false ]
                      write ]
                    [ edge "salt" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            let shotCode =
                graph
                    [ node "shot" "ShotNoise"
                        [ p "availableMemory" "2048" false
                          p "type" "Float32" false
                          p "width" "12" false
                          p "height" "13" false
                          p "depth" "4" false
                          p "scale" "2.0" false ]
                      write ]
                    [ edge "shot" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            let speckleCode =
                graph
                    [ node "speckle" "SpeckleNoise"
                        [ p "availableMemory" "2048" false
                          p "type" "Float32" false
                          p "width" "12" false
                          p "height" "13" false
                          p "depth" "4" false
                          p "std" "0.5" false ]
                      write ]
                    [ edge "speckle" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains saltCode "|> saltAndPepperNoise<float32> 12u 13u 4u 0.01" "SaltAndPepperNoise should lower with probability."
            Expect.stringContains shotCode "|> shotNoise<float32> 12u 13u 4u 2.0" "ShotNoise should lower with scale."
            Expect.stringContains speckleCode "|> speckleNoise<float32> 12u 13u 4u 0.5" "SpeckleNoise should lower with std."

        testCase "coordinate sources lower to Float64 coordinate image streams" <| fun _ ->
            let write =
                node "write" "Write"
                    [ p "output" "coords" false
                      p "suffix" ".tiff" false ]

            let codeFor id functionId =
                let coord =
                    node id functionId
                        [ p "availableMemory" "2048" false
                          p "width" "12" false
                          p "height" "13" false
                          p "depth" "4" false ]

                graph [ coord; write ] [ edge id "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains (codeFor "x" "CoordinateX") "|> coordinateX 12u 13u 4u" "CoordinateX should generate a coordinate source."
            Expect.stringContains (codeFor "y" "CoordinateY") "|> coordinateY 12u 13u 4u" "CoordinateY should generate a coordinate source."
            Expect.stringContains (codeFor "z" "CoordinateZ") "|> coordinateZ 12u 13u 4u" "CoordinateZ should generate a coordinate source."

        testCase "bias model reducer can feed correction stage" <| fun _ ->
            let sample =
                node "sample" "ReadRandom"
                    [ p "availableMemory" "4096" false
                      p "type" "Float64" false
                      p "depth" "8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let fit =
                node "fit" "FitBiasModel"
                    [ p "type" "Float64" false
                      p "order" "2" false
                      p "depth" "32" false ]

            let read =
                node "read" "Read"
                    [ p "availableMemory" "4096" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let correct =
                node "correct" "CorrectBias"
                    [ p "type" "Float64" false
                      p "model" "biasModel" true ]

            let write =
                node "write" "Write"
                    [ p "output" "corrected" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ sample; fit; read; correct; write ]
                    [ edge "sample" "output" 0 "fit" "input" 0
                      edge "fit" "reducerOutput" 0 "correct" "parameterInput" 1
                      edge "read" "output" 0 "correct" "input" 0
                      edge "correct" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> readRandom<float> 8u \"input\" \".tiff\"" "Bias fit should be able to use readRandom."
            Expect.stringContains code ">=> fitBiasModel<float> 2 32u" "FitBiasModel should lower to the polynomial reducer."
            Expect.stringContains code "|> drain" "Linked bias model reducer should be drained into a binding."
            Expect.stringContains code ">=> correctBias<float> FitBiasModel0" "CorrectBias should receive the linked bias model binding."

        testCase "serial section manifest reducer can feed slicewise transform application" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "4096" false
                      p "type" "Float64" false
                      p "input" "sections" false
                      p "suffix" ".tiff" false ]

            let manifest =
                node "manifest" "SerialImageTranslationManifest"
                    [ p "type" "Float64" false
                      p "maxShift" "4" false ]

            let apply =
                node "apply" "SerialApplyManifestInBoundingBox"
                    [ p "type" "Float64" false
                      p "manifest" "serialManifest" true
                      p "background" "0.0" false ]

            let write =
                node "write" "Write"
                    [ p "output" "aligned" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; manifest; apply; write ]
                    [ edge "read" "output" 0 "manifest" "input" 0
                      edge "manifest" "reducerOutput" 0 "apply" "parameterInput" 1
                      edge "read" "output" 0 "apply" "input" 0
                      edge "apply" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> serialImageTranslationManifest<float> 4" "SerialImageTranslationManifest should lower to the serial reducer."
            Expect.stringContains code ">=> serialApplyManifestInBoundingBox<float> SerialImageTranslationManifest0 0.0" "Serial apply should receive the linked manifest binding."

        testCase "noise add-stage boxes lower to streaming filters" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "2048" false
                      p "type" "Float32" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let normal = node "normal" "AddNormalNoise" [ p "type" "Float32" false; p "mean" "1.0" false; p "std" "0.25" false ]
            let salt = node "salt" "AddSaltAndPepperNoise" [ p "type" "Float32" false; p "probability" "0.01" false ]
            let shot = node "shot" "AddShotNoise" [ p "type" "Float32" false; p "scale" "2.0" false ]
            let speckle = node "speckle" "AddSpeckleNoise" [ p "type" "Float32" false; p "std" "0.5" false ]
            let write = node "write" "Write" [ p "output" "noise" false; p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; normal; salt; shot; speckle; write ]
                    [ edge "read" "output" 0 "normal" "input" 0
                      edge "normal" "output" 0 "salt" "input" 0
                      edge "salt" "output" 0 "shot" "input" 0
                      edge "shot" "output" 0 "speckle" "input" 0
                      edge "speckle" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> addNormalNoise 1.0 0.25" "AddNormalNoise should lower with mean and std."
            Expect.stringContains code ">=> addSaltAndPepperNoise 0.01" "AddSaltAndPepperNoise should lower with probability."
            Expect.stringContains code ">=> addShotNoise 2.0" "AddShotNoise should lower with scale."
            Expect.stringContains code ">=> addSpeckleNoise 0.5" "AddSpeckleNoise should lower with std."

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

        testCase "readRange box lowers to ranged DSL source" <| fun _ ->
            let read =
                node "read" "ReadRange"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt16" false
                      p "first" "1" false
                      p "step" "2" false
                      p "last" "end-1" false
                      p "input" "input" false
                      p "suffix" ".tif" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph [ read; write ] [ edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> readRange<uint16> \"1\" 2 \"end-1\" \"input\" \".tif\"" "ReadRange should generate a typed clamped range reader."

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

        testCase "nexus boxes lower to nexus DSL functions" <| fun _ ->
            let info =
                node "info" "GetNexusInfo"
                    [ p "input" "scan.h5" false
                      p "datasetPath" "/entry/data/data" false
                      p "frameAxis" "0" false
                      p "yAxis" "1" false
                      p "xAxis" "2" false ]

            let read =
                node "read" "ReadNexusSlab"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt16" false
                      p "input" "scan.h5" false
                      p "datasetPath" "/entry/data/data" false
                      p "slabDepth" "8" false
                      p "frameAxis" "0" false
                      p "yAxis" "1" false
                      p "xAxis" "2" false ]

            let write =
                node "write" "WriteZarr"
                    [ p "output" "output.zarr" false
                      p "name" "converted" false
                      p "depth" "" true
                      p "chunkX" "16" false
                      p "chunkY" "17" false
                      p "chunkZ" "8" false
                      p "physicalSizeX" "1.0" false
                      p "physicalSizeY" "1.0" false
                      p "physicalSizeZ" "1.0" false
                      p "maxConcurrentWrites" "0" false ]

            let code =
                graph
                    [ info; read; write ]
                    [ edge "read" "output" 0 "write" "input" 0
                      edge "info" "reducerOutput" 9 "write" "parameterInput" 2 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let ChunkInfo0 = getNexusInfo \"scan.h5\" \"/entry/data/data\" 0 1 2" "GetNexusInfo should generate a metadata binding."
            Expect.stringContains code "|> readNexusSlab<uint16> \"scan.h5\" \"/entry/data/data\" 8u 0 1 2" "ReadNexusSlab should generate the NeXus slab reader."
            Expect.stringContains code ">=> writeZarr \"output.zarr\" \"converted\" ChunkInfo0.size[2] 16u 17u 8u 1.0 1.0 1.0 0" "WriteZarr should accept linked NeXus metadata."

        testCase "write nexus box lowers to writeNexus" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt16" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let write =
                node "write" "WriteNexus"
                    [ p "output" "output.h5" false
                      p "datasetPath" "/entry/data/data" false
                      p "depth" "32" false
                      p "chunkX" "16" false
                      p "chunkY" "17" false
                      p "chunkZ" "8" false
                      p "frameAxis" "0" false
                      p "yAxis" "1" false
                      p "xAxis" "2" false ]

            let code =
                graph [ read; write ] [ edge "read" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> writeNexus \"output.h5\" \"/entry/data/data\" 32u 16u 17u 8u 0 1 2" "WriteNexus should generate the NeXus writer."
            Expect.stringContains code "|> sink" "Terminal WriteNexus should be sunk."

        testCase "axis-aligned resize and resample boxes lower to plan functions" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "Float32" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let resize =
                node "resize" "Resize"
                    [ p "type" "Float32" false
                      p "width" "20" false
                      p "height" "21" false
                      p "depth" "22" false
                      p "interpolation" "NearestNeighbor" false ]

            let resample =
                node "resample" "Resample"
                    [ p "type" "Float32" false
                      p "factorX" "0.5" false
                      p "factorY" "2.0" false
                      p "factorZ" "1.5" false
                      p "interpolation" "Linear" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; resize; resample; write ]
                    [ edge "read" "output" 0 "resize" "input" 0
                      edge "resize" "output" 0 "resample" "input" 0
                      edge "resample" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> resize<float32> 20u 21u 22u \"NearestNeighbor\"" "Resize should generate the explicit-size resampler."
            Expect.stringContains code "|> resample<float32> 0.5 2.0 1.5 \"Linear\"" "Resample should generate the factor resampler."

        testCase "padding and crop boxes lower to streaming geometry DSL functions" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let padding =
                node "pad" "CreatePadding"
                    [ p "type" "UInt8" false
                      p "beforeX" "1" false
                      p "afterX" "2" false
                      p "beforeY" "3" false
                      p "afterY" "4" false
                      p "beforeZ" "5" false
                      p "afterZ" "6" false
                      p "value" "7.0" false ]

            let crop =
                node "crop" "Crop"
                    [ p "type" "UInt8" false
                      p "beforeX" "1" false
                      p "afterX" "1" false
                      p "beforeY" "2" false
                      p "afterY" "2" false
                      p "beforeZ" "3" false
                      p "afterZ" "3" false ]

            let write =
                node "write" "Write"
                    [ p "output" "output" false
                      p "suffix" ".tiff" false ]

            let code =
                graph [ read; padding; crop; write ]
                    [ edge "read" "output" 0 "pad" "input" 0
                      edge "pad" "output" 0 "crop" "input" 0
                      edge "crop" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> createPadding<uint8> 1u 2u 3u 4u 5u 6u 7.0" "CreatePadding should generate a typed six-side padding stage."
            Expect.stringContains code ">=> crop<uint8> 1u 1u 2u 2u 3u 3u" "Crop should generate a typed six-side crop stage."

        testCase "marchingCubes and writeMesh boxes lower to streaming mesh DSL functions" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let marching =
                node "mesh" "MarchingCubes"
                    [ p "type" "UInt8" false
                      p "surfaceValue" "1.0" false ]

            let write =
                node "writeMesh" "WriteMesh"
                    [ p "output" "surface.obj" false
                      p "format" "auto" false ]

            let code =
                graph [ read; marching; write ]
                    [ edge "read" "output" 0 "mesh" "input" 0
                      edge "mesh" "output" 0 "writeMesh" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> marchingCubes<uint8> 1.0" "MarchingCubes should generate a typed streaming mesh stage."
            Expect.stringContains code ">=> writeMesh \"surface.obj\" \"auto\"" "WriteMesh should generate the mesh writer."
            Expect.stringContains code "|> sink" "Terminal WriteMesh should run the mesh writer."

            let surfaceArea =
                node "area" "SurfaceArea"
                    [ p "xUnit" "2.0" false
                      p "yUnit" "3.0" false
                      p "zUnit" "4.0" false ]

            let areaCode =
                graph [ read; marching; surfaceArea ]
                    [ edge "read" "output" 0 "mesh" "input" 0
                      edge "mesh" "output" 0 "area" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains areaCode ">=> surfaceArea 2.0 3.0 4.0" "SurfaceArea should generate the physical area reducer."
            Expect.stringContains areaCode "|> drain" "Terminal SurfaceArea should be drained."

        testCase "dogKeypoints and point-set IO boxes lower to CSV point-set DSL functions" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let dog =
                node "dog" "DogKeypoints"
                    [ p "type" "Float64" false
                      p "sigma0" "0.5" false
                      p "scaleFactor" "1.2" false
                      p "scaleLevels" "4" false
                      p "contrastThreshold" "0.001" false
                      p "stride" "3" false ]

            let write =
                node "writePoints" "WritePointSet"
                    [ p "output" "points" false
                      p "suffix" ".csv" false ]

            let code =
                graph [ read; dog; write ]
                    [ edge "read" "output" 0 "dog" "input" 0
                      edge "dog" "output" 0 "writePoints" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> dogKeypoints<float> 0.5 1.2 4u 0.001 3u" "DogKeypoints should generate a typed point detector."
            Expect.stringContains code ">=> writePointSet \"points\" \".csv\"" "WritePointSet should generate the point writer with suffix."
            Expect.stringContains code "|> sink" "Terminal WritePointSet should run the point writer."

            let sift =
                node "sift" "SiftKeypoints"
                    [ p "type" "Float64" false
                      p "sigma0" "0.5" false
                      p "scaleFactor" "1.2" false
                      p "scaleLevels" "4" false
                      p "contrastThreshold" "0.001" false
                      p "stride" "3" false ]

            let siftCode =
                graph [ read; sift; write ]
                    [ edge "read" "output" 0 "sift" "input" 0
                      edge "sift" "output" 0 "writePoints" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains siftCode ">=> siftKeypoints<float> 0.5 1.2 4u 0.001 3u" "SiftKeypoints should generate a typed point detector."

            let logBlob =
                node "log" "LogBlobKeypoints"
                    [ p "type" "Float64" false
                      p "sigma" "1.5" false
                      p "threshold" "0.02" false
                      p "stride" "2" false ]

            let hessian =
                node "hessian" "HessianKeypoints"
                    [ p "type" "Float64" false
                      p "sigma" "1.5" false
                      p "responseKind" "Tube" false
                      p "threshold" "0.02" false
                      p "stride" "2" false ]

            let harris =
                node "harris" "Harris3DKeypoints"
                    [ p "type" "Float64" false
                      p "sigma" "1.0" false
                      p "rho" "1.5" false
                      p "k" "0.04" false
                      p "threshold" "0.02" false
                      p "stride" "2" false ]

            let forstner =
                node "forstner" "Forstner3DKeypoints"
                    [ p "type" "Float64" false
                      p "sigma" "1.0" false
                      p "rho" "1.5" false
                      p "threshold" "0.02" false
                      p "stride" "2" false ]

            let phase =
                node "phase" "PhaseCongruencyKeypoints"
                    [ p "type" "Float64" false
                      p "sigma" "1.5" false
                      p "threshold" "0.02" false
                      p "stride" "2" false ]

            let generated detector expected message =
                graph [ read; detector; write ]
                    [ edge "read" "output" 0 detector.Id "input" 0
                      edge detector.Id "output" 0 "writePoints" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph
                |> fun code -> Expect.stringContains code expected message

            generated logBlob ">=> logBlobKeypoints<float> 1.5 0.02 2u" "LogBlobKeypoints should generate a typed local blob detector."
            generated hessian ">=> hessianKeypoints<float> 1.5 \"Tube\" 0.02 2u" "HessianKeypoints should generate a typed local Hessian detector."
            generated harris ">=> harris3DKeypoints<float> 1.0 1.5 0.04 0.02 2u" "Harris3DKeypoints should generate a typed local Harris detector."
            generated forstner ">=> forstner3DKeypoints<float> 1.0 1.5 0.02 2u" "Forstner3DKeypoints should generate a typed local Förstner detector."
            generated phase ">=> phaseCongruencyKeypoints<float> 1.5 0.02 2u" "PhaseCongruencyKeypoints should generate a typed local phase detector."

            let readPoints =
                node "readPoints" "ReadPointSet"
                    [ p "availableMemory" "1024" false
                      p "input" "points.csv" false ]

            let readPointCode =
                graph [ readPoints ] []
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains readPointCode "debug 1u 1024UL" "ReadPointSet should include a debug source."
            Expect.stringContains readPointCode "|> readPointSet \"points.csv\"" "ReadPointSet should generate the CSV point reader."

            let distances =
                node "distances" "PointPairDistances"
                    [ p "xUnit" "2.0" false
                      p "yUnit" "3.0" false
                      p "zUnit" "4.0" false ]

            let distanceCode =
                graph [ read; dog; distances ]
                    [ edge "read" "output" 0 "dog" "input" 0
                      edge "dog" "output" 0 "distances" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains distanceCode ">=> pointPairDistances 2.0 3.0 4.0" "PointPairDistances should generate the physical distance matrix reducer."
            Expect.stringContains distanceCode "|> drain" "Terminal PointPairDistances should be drained."

            let writeMatrix =
                node "writeMatrix" "WriteMatrix"
                    [ p "output" "distances" false
                      p "suffix" ".csv" false ]

            let distanceWriteCode =
                graph [ read; dog; distances; writeMatrix ]
                    [ edge "read" "output" 0 "dog" "input" 0
                      edge "dog" "output" 0 "distances" "input" 0
                      edge "distances" "reducerOutput" 0 "writeMatrix" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains distanceWriteCode ">=> pointPairDistances 2.0 3.0 4.0" "PointPairDistances should feed matrix CSV writing."
            Expect.stringContains distanceWriteCode ">=> writeMatrix \"distances\" \".csv\"" "WriteMatrix should generate the suffix-controlled matrix writer."
            Expect.stringContains distanceWriteCode "|> sink" "Terminal WriteMatrix should run the matrix writer."

            let writeCSVMatrix =
                node "writeCSVMatrix" "WriteCSV"
                    [ p "output" "distances" false
                      p "dataKind" "Matrix" false ]

            let csvMatrixCode =
                graph [ read; dog; distances; writeCSVMatrix ]
                    [ edge "read" "output" 0 "dog" "input" 0
                      edge "dog" "output" 0 "distances" "input" 0
                      edge "distances" "reducerOutput" 0 "writeCSVMatrix" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains csvMatrixCode ">=> writeCSVMatrix \"distances\"" "WriteCSV should lower matrix input to the matrix CSV writer."
            Expect.stringContains csvMatrixCode "|> sink" "Terminal WriteCSV should run the selected CSV writer."

            let writeCSVPoints =
                node "writeCSVPoints" "WriteCSV"
                    [ p "output" "points" false
                      p "dataKind" "PointSet" false ]

            let csvPointCode =
                graph [ read; dog; writeCSVPoints ]
                    [ edge "read" "output" 0 "dog" "input" 0
                      edge "dog" "output" 0 "writeCSVPoints" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains csvPointCode ">=> writeCSVPointSet \"points\"" "WriteCSV should lower point-set input to the point-set CSV writer."

            let fixedPoints =
                node "fixedPoints" "ReadPointSet"
                    [ p "availableMemory" "1024" false
                      p "input" "fixed.csv" false ]

            let movingPoints =
                node "movingPoints" "ReadPointSet"
                    [ p "availableMemory" "1024" false
                      p "input" "moving.csv" false ]

            let registration =
                node "registration" "AffineRegistration"
                    [ p "maxIterations" "5" false
                      p "initialLinearStep" "0.05" false
                      p "initialTranslationStep" "1.0" false
                      p "minStep" "0.0001" false
                      p "stepShrink" "0.5" false ]

            let writeTransform =
                node "writeTransform" "WriteMatrix"
                    [ p "output" "transform" false
                      p "suffix" ".csv" false ]

            let registrationCode =
                graph [ fixedPoints; movingPoints; registration; writeTransform ]
                    [ edge "fixedPoints" "output" 0 "registration" "input" 0
                      edge "movingPoints" "output" 0 "registration" "input" 1
                      edge "registration" "reducerOutput" 1 "writeTransform" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains registrationCode ">=> affineRegistrationMatrices" "AffineRegistration should lower to the point-set registration matrix stage."
            Expect.stringContains registrationCode ">=> selectGroupedValueOutput 2u 1u" "Connecting the inverse transform output should select the second matrix."
            Expect.stringContains registrationCode ">=> writeMatrix \"transform\" \".csv\"" "AffineRegistration matrices should be writable with writeMatrix."

        testCase "streamed object and painter boxes lower to object-mask DSL stages" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1073741824" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let objects =
                node "objects" "StreamConnectedObjects"
                    [ p "connectivity" "TwentySix" false ]

            let paint =
                node "paint" "PaintObjects"
                    [ p "width" "64" false
                      p "height" "48" false ]

            let write =
                node "write" "Write"
                    [ p "output" "mask" false
                      p "suffix" ".tiff" false ]

            let code =
                graph [ read; objects; paint; write ]
                    [ edge "read" "output" 0 "objects" "input" 0
                      edge "objects" "output" 0 "paint" "input" 0
                      edge "paint" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> streamConnectedObjects<uint8> ObjectConnectivity.TwentySix" "StreamConnectedObjects should generate the connectivity-aware object streamer."
            Expect.stringContains code ">=> paintObjects 64u 48u" "PaintObjects should generate the UInt8 mask painter."
            Expect.stringContains code ">=> write \"mask\" \".tiff\"" "Painted masks should connect to normal image writing."

            let croppedPaint =
                node "croppedPaint" "PaintObjectsCropped" []

            let croppedCode =
                graph [ read; objects; croppedPaint ]
                    [ edge "read" "output" 0 "objects" "input" 0
                      edge "objects" "output" 0 "croppedPaint" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains croppedCode ">=> paintObjectsCropped" "PaintObjectsCropped should generate the minimal-mask painter."

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

            Expect.stringContains code "let Float641 = (System.Math.Cos Float640)" "ScalarFunction should lower F# function selection."
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

            let relabel =
                node "relabel" "RelabelComponents"
                    [ p "minimumObjectSize" "10" false
                      p "windowSize" "5" false ]
            let signed =
                node "signed" "SignedDistanceBand"
                    [ p "bandRadius" "9" false
                      p "stride" "5" false ]
            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; relabel; signed; write ]
                    [ edge "read" "output" 0 "relabel" "input" 0
                      edge "relabel" "output" 0 "signed" "input" 0
                      edge "signed" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> relabelComponents 10u 5u" "relabelComponents should lower with size threshold and window size."
            Expect.stringContains code ">=> signedDistanceBand 9u 5u" "signedDistanceBand should lower with band radius and stride."

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

        testCase "image op image shares a common source used directly and through a branch" <| fun _ ->
            let read =
                node "read" "ReadVolume"
                    [ p "availableMemory" "1073741824" false
                      p "type" "Float32" false
                      p "input" "volumedata.tif" false ]

            let correct =
                node "correct" "SerialPolynomialBiasCorrect"
                    [ p "type" "Float32" false
                      p "order" "2" false ]

            let op =
                node "op" "ImageOpImage"
                    [ p "operation" "-" false
                      p "type" "Float32" false ]

            let write =
                node "write" "Write"
                    [ p "output" "volumedata" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; correct; op; write ]
                    [ edge "read" "dataOutput" 0 "correct" "dataInput" 0
                      edge "read" "dataOutput" 0 "op" "dataInput" 0
                      edge "correct" "dataOutput" 0 "op" "dataInput" 1
                      edge "op" "dataOutput" 0 "write" "dataInput" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.equal (code.Split("|> readVolume<float32>").Length - 1) 1 "The shared readVolume source should be generated once."
            Expect.stringContains code ">=>> (identity" "The direct branch should use the public identity stage in shared fan-out."
            Expect.stringContains code "serialPolynomialBiasCorrect<float32> 2" "The corrected branch should still include the bias correction stage."
            Expect.stringContains code ">>=> subPair" "The two branches should be combined by the pair operation."

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

        testCase "high-value SimpleITK filter boxes lower to StackProcessing stages" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let clamp =
                node "clamp" "Clamp"
                    [ p "type" "Float64" false
                      p "lower" "0.0" false
                      p "upper" "1.0" false ]

            let median =
                node "median" "SmoothWMedian"
                    [ p "type" "Float64" false
                      p "radius" "1" false
                      p "windowSize" "3" false ]

            let bilateral =
                node "bilateral" "SmoothWBilateral"
                    [ p "type" "Float64" false
                      p "domainSigma" "2.0" false
                      p "rangeSigma" "10.0" false
                      p "windowSize" "5" false ]

            let smooth =
                node "smooth" "SmoothWGauss"
                    [ p "sigma" "1.0" false
                      p "outputRegionMode" "None" false
                      p "boundaryCondition" "None" false
                      p "windowSize" "None" false ]

            let edgeFilter =
                node "edge" "GradientMagnitude"
                    [ p "type" "Float64" false
                      p "windowSize" "5" false ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; clamp; median; bilateral; smooth; edgeFilter; write ]
                    [ edge "read" "output" 0 "clamp" "input" 0
                      edge "clamp" "output" 0 "median" "input" 0
                      edge "median" "output" 0 "bilateral" "input" 0
                      edge "bilateral" "output" 0 "smooth" "input" 0
                      edge "smooth" "output" 0 "edge" "input" 0
                      edge "edge" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> clamp<float> 0.0 1.0" "Clamp should lower with type and bounds."
            Expect.stringContains code ">=> smoothWMedian<float> 1u 3u" "smoothWMedian should lower with radius and window size."
            Expect.stringContains code ">=> smoothWBilateral<float> 2.0 10.0 5u" "smoothWBilateral should lower with sigmas and window size."
            Expect.stringContains code ">=> smoothWGauss 1.0 None None None" "smoothWGauss should lower with explicit options."
            Expect.stringContains code ">=> gradientMagnitude<float> 5u" "Gradient magnitude should lower with its window size."

        testCase "sum projection lowers to a Float64 image reducer" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt16" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let projection =
                node "projection" "SumProjection"
                    [ p "type" "UInt16" false
                      p "function" "Log1pAbs" false ]

            let show =
                node "show" "ShowImage" []

            let code =
                graph
                    [ read; projection; show ]
                    [ edge "read" "output" 0 "projection" "input" 0
                      edge "projection" "output" 0 "show" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> sumProjection<uint16> \"Log1pAbs\"" "SumProjection should lower with input pixel type and pre-sum transform."
            Expect.stringContains code ">=> show showImagePlot" "The projection output should be connectable to showImage."

        testCase "comparison boxes lower to pair stages" <| fun _ ->
            let left =
                node "left" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "left" false
                      p "suffix" ".tiff" false ]
            let right =
                node "right" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "right" false
                      p "suffix" ".tiff" false ]
            let comparison =
                node "cmp" "ImageComparison"
                    [ p "operation" ">=" false
                      p "type" "Float64" false ]
            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ left; right; comparison; write ]
                    [ edge "left" "output" 0 "cmp" "I" 0
                      edge "right" "output" 0 "cmp" "J" 1
                      edge "cmp" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> greaterEqual<float>" "Image comparison should lower to the selected typed pair stage."

        testCase "vector image boxes lower to StackProcessing vector stages" <| fun _ ->
            let read id input =
                node id "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" input false
                      p "suffix" ".tiff" false ]

            let write = node "write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ]

            let vectorCode =
                graph
                    [ read "x" "x"; read "y" "y"; read "z" "z"
                      node "toVector" "ToVectorImage" []
                      node "append" "AppendVectorElement" []
                      node "map" "VectorMapElements" [ p "function" "sqrt" false ]
                      node "element" "VectorElement" [ p "component" "2" false ]
                      write ]
                    [ edge "x" "output" 0 "toVector" "I" 0
                      edge "y" "output" 0 "toVector" "J" 1
                      edge "toVector" "output" 0 "append" "Vector" 0
                      edge "z" "output" 0 "append" "Float64" 1
                      edge "append" "output" 0 "map" "input" 0
                      edge "map" "output" 0 "element" "input" 0
                      edge "element" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains vectorCode ">=> toVectorImage<float>" "toVectorImage should lower to the vector composition pair stage."
            Expect.stringContains vectorCode ">=> appendVectorElement" "appendVectorElement should lower as a mixed vector/scalar pair stage."
            Expect.stringContains vectorCode ">=> vectorMapElements \"sqrt\"" "VectorMapElements should lower with the selected function."
            Expect.stringContains vectorCode ">=> vectorElement<float> 2" "VectorElement should lower with the selected component."

            let gradientCode =
                graph
                    [ read "source" "source"
                      node "gradient" "Gradient" [ p "order" "1" false; p "windowSize" "5" false ]
                      node "element" "VectorElement" [ p "component" "0" false ]
                      write ]
                    [ edge "source" "output" 0 "gradient" "input" 0
                      edge "gradient" "output" 0 "element" "input" 0
                      edge "element" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains gradientCode ">=> gradient 1u (Some 5u)" "Gradient should lower with order and window size."

            let angleCode =
                graph
                    [ read "x" "x"; read "y" "y"
                      node "toVector" "ToVectorImage" []
                      node "angle" "VectorAngleTo" [ p "x" "0.0" false; p "y" "1.0" false; p "z" "0.0" false ]
                      write ]
                    [ edge "x" "output" 0 "toVector" "I" 0
                      edge "y" "output" 0 "toVector" "J" 1
                      edge "toVector" "output" 0 "angle" "input" 0
                      edge "angle" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains angleCode ">=> vectorAngleTo [ 0.0; 1.0; 0.0 ]" "VectorAngleTo should lower with the reference vector."

            let structureTensorCode =
                graph
                    [ read "source" "source"
                      node "tensor" "StructureTensor" [ p "sigma" "1.0" false; p "rho" "2.0" false ]
                      write ]
                    [ edge "source" "output" 0 "tensor" "input" 0
                      edge "tensor" "output" 2 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains structureTensorCode ">=> structureTensor 1.0 2.0" "StructureTensor should lower with sigma and rho."
            Expect.stringContains structureTensorCode ">=> selectGroupedOutput 4u 2u" "Connecting a StructureTensor output port should select the corresponding 3-vector stream."

            let pcaCode =
                graph
                    [ read "x" "x"; read "y" "y"
                      node "toVector" "ToVectorImage" []
                      node "pca" "PCA" [ p "components" "3" false ]
                      write ]
                    [ edge "x" "output" 0 "toVector" "I" 0
                      edge "y" "output" 0 "toVector" "J" 1
                      edge "toVector" "output" 0 "pca" "input" 0
                      edge "pca" "output" 1 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains pcaCode ">=> PCA 3u" "PCA should lower as a reducer stage with component count."
            Expect.stringContains pcaCode ">=> selectGroupedOutput (3u + 1u) 1u" "Connecting a PCA output port should select the corresponding eigensystem stream."

            let dotCode =
                graph
                    [ read "x" "x"; read "y" "y"; read "u" "u"; read "v" "v"
                      node "leftVector" "ToVectorImage" []
                      node "rightVector" "ToVectorImage" []
                      node "dot" "VectorDot" []
                      write ]
                    [ edge "x" "output" 0 "leftVector" "I" 0
                      edge "y" "output" 0 "leftVector" "J" 1
                      edge "u" "output" 0 "rightVector" "I" 0
                      edge "v" "output" 0 "rightVector" "J" 1
                      edge "leftVector" "output" 0 "dot" "U" 0
                      edge "rightVector" "output" 0 "dot" "V" 1
                      edge "dot" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains dotCode ">=> vectorDot" "VectorDot should lower to the pixelwise dot-product stage."

            let crossCode =
                graph
                    [ read "x" "x"; read "y" "y"; read "u" "u"; read "v" "v"
                      node "leftVector" "ToVectorImage" []
                      node "rightVector" "ToVectorImage" []
                      node "cross" "VectorCross3D" []
                      node "element" "VectorElement" [ p "component" "0" false ]
                      write ]
                    [ edge "x" "output" 0 "leftVector" "I" 0
                      edge "y" "output" 0 "leftVector" "J" 1
                      edge "u" "output" 0 "rightVector" "I" 0
                      edge "v" "output" 0 "rightVector" "J" 1
                      edge "leftVector" "output" 0 "cross" "U" 0
                      edge "rightVector" "output" 0 "cross" "V" 1
                      edge "cross" "output" 0 "element" "input" 0
                      edge "element" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains crossCode ">=> vectorCross3D" "VectorCross3D should lower to the pixelwise 3D cross-product stage."

        testCase "complex image boxes lower to StackProcessing complex stages" <| fun _ ->
            let read id input =
                node id "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" input false
                      p "suffix" ".tiff" false ]

            let write = node "write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ]

            let complexCode =
                graph
                    [ read "real" "real"; read "imag" "imag"
                      node "complex" "ComplexFromReIm" []
                      node "conjugate" "ComplexConjugate" []
                      node "modulus" "ComplexModulus" []
                      write ]
                    [ edge "real" "output" 0 "complex" "Re" 0
                      edge "imag" "output" 0 "complex" "Im" 1
                      edge "complex" "output" 0 "conjugate" "input" 0
                      edge "conjugate" "output" 0 "modulus" "input" 0
                      edge "modulus" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains complexCode ">=> toComplex" "ComplexFromReIm should lower to the real/imag composition stage."
            Expect.stringContains complexCode ">=> conjugate" "ComplexConjugate should lower to conjugate."
            Expect.stringContains complexCode ">=> modulus" "ComplexModulus should lower to modulus."

            let polarCode =
                graph
                    [ read "magnitude" "magnitude"; read "phase" "phase"
                      node "complex" "ComplexPolar" []
                      node "re" "ComplexRe" []
                      write ]
                    [ edge "magnitude" "output" 0 "complex" "Modulus" 0
                      edge "phase" "output" 0 "complex" "Arg" 1
                      edge "complex" "output" 0 "re" "input" 0
                      edge "re" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains polarCode ">=> polarToComplex" "ComplexPolar should lower to polarToComplex."
            Expect.stringContains polarCode ">=> Re" "ComplexRe should lower to Re."

            let imCode =
                graph
                    [ read "real" "real"; read "imag" "imag"
                      node "complex" "ComplexFromReIm" []
                      node "im" "ComplexIm" []
                      write ]
                    [ edge "real" "output" 0 "complex" "Re" 0
                      edge "imag" "output" 0 "complex" "Im" 1
                      edge "complex" "output" 0 "im" "input" 0
                      edge "im" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains imCode ">=> Im" "ComplexIm should lower to Im."

            let argCode =
                graph
                    [ read "real" "real"; read "imag" "imag"
                      node "complex" "ComplexFromReIm" []
                      node "arg" "ComplexArg" []
                      write ]
                    [ edge "real" "output" 0 "complex" "Re" 0
                      edge "imag" "output" 0 "complex" "Im" 1
                      edge "complex" "output" 0 "arg" "input" 0
                      edge "arg" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains argCode ">=> arg" "ComplexArg should lower to arg."

        testCase "fourier boxes lower to chunk-backed StackProcessing stages" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]
            let fft =
                node "fft" "FFT"
                    [ p "type" "Float64" false
                      p "chunkX" "8" false
                      p "chunkY" "9" false
                      p "chunkZ" "2" false ]
            let shift =
                node "shift" "ShiftFFT"
                    [ p "chunkX" "8" false
                      p "chunkY" "9" false
                      p "chunkZ" "2" false ]
            let inv =
                node "inv" "InvFFT"
                    [ p "chunkX" "8" false
                      p "chunkY" "9" false
                      p "chunkZ" "2" false ]
            let write = node "write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; fft; shift; inv; write ]
                    [ edge "read" "output" 0 "fft" "input" 0
                      edge "fft" "output" 0 "shift" "input" 0
                      edge "shift" "output" 0 "inv" "input" 0
                      edge "inv" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> FFT<float> 8u 9u 2u" "FFT should lower with pixel type and chunk shape."
            Expect.stringContains code ">=> shiftFFT 8u 9u 2u" "ShiftFFT should lower with chunk shape."
            Expect.stringContains code ">=> invFFT 8u 9u 2u" "InvFFT should lower with chunk shape."

        testCase "morphology and streaming label additions lower to StackProcessing stages" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]
            let gray =
                node "gray" "GrayscaleErode"
                    [ p "type" "Float64" false
                      p "radius" "2" false
                      p "windowSize" "5" false ]
            let contour =
                node "contour" "BinaryContour"
                    [ p "fullyConnected" "true" false
                      p "windowSize" "3" false ]
            let removeSmall =
                node "removeSmall" "RemoveSmallObjects"
                    [ p "maximumVolume" "11" false
                      p "connectivity" "TwentySix" false ]
            let fillSmall =
                node "fillSmall" "FillSmallHoles"
                    [ p "maximumVolume" "13" false
                      p "connectivity" "Six" false ]
            let code =
                graph
                    [ read; gray; contour; removeSmall; fillSmall ]
                    [ edge "read" "output" 0 "gray" "input" 0
                      edge "gray" "output" 0 "contour" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> grayscaleErode<float> 2u 5u" "Grayscale morphology should lower with type, radius, and window size."
            Expect.stringContains code ">=> binaryContour true 3u" "Binary contour should lower with connectivity and window size."
            Expect.stringContains code ">=> removeSmallObjects 11UL ObjectConnectivity.TwentySix" "removeSmallObjects should lower with maximum volume and connectivity."
            Expect.stringContains code ">=> fillSmallHoles 13UL ObjectConnectivity.Six" "fillSmallHoles should lower with maximum volume and connectivity."

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

        testCase "estimateHistogram source exposes histogram map and diagnostics" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let estimate =
                node "estimate" "EstimateHistogram"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "slices" "12" false
                      p "input" "input" false
                      p "suffix" ".tiff" false
                      p "down" "4" false
                      p "estimator" "DKWAndHoldout" false
                      p "confidence" "0.95" false ]

            let equalize =
                node "equalize" "HistogramEqualization"
                    [ p "type" "UInt8" false
                      p "histogram" "" true ]

            let print =
                node "print" "Print"
                    [ p "format" "n={input1}, eps={input2}" false
                      p "input1" "" true
                      p "input2" "" true ]

            let code =
                graph
                    [ read; estimate; equalize; print ]
                    [ edge "estimate" "reducerOutput" 0 "equalize" "parameterInput" 1
                      edge "estimate" "reducerOutput" 1 "print" "parameterInput" 1
                      edge "estimate" "reducerOutput" 2 "print" "parameterInput" 2
                      edge "read" "output" 0 "equalize" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "|> estimateHistogram<uint8> 12u \"input\" \".tiff\" 4u \"DKWAndHoldout\" 0.95" "EstimateHistogram should lower to the typed random-sampling source."
            Expect.stringContains code "let Histogram0 =" "Linked estimate outputs should be bound once."
            Expect.stringContains code ">=> histogramEqualization<uint8> Histogram0.Histogram" "The map output should feed histogram-based image stages."
            Expect.stringContains code "n={Histogram0.Samples}, eps={Histogram0.CdfHalfWidth}" "Diagnostics should be addressable as scalar-like outputs."

        testCase "writeCSV can write histogram data" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt8" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let histogram =
                node "histogram" "HistogramData" []

            let write =
                node "writeHistogram" "WriteCSV"
                    [ p "output" "histogram" false
                      p "dataKind" "Histogram" false ]

            let code =
                graph
                    [ read; histogram; write ]
                    [ edge "read" "output" 0 "histogram" "input" 0
                      edge "histogram" "reducerOutput" 0 "writeHistogram" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> histogram ()" "HistogramData should still generate the histogram reducer."
            Expect.stringContains code ">=> writeCSVHistogram \"histogram\"" "WriteCSV should lower histogram input to the histogram CSV writer."
            Expect.stringContains code "|> sink" "Terminal histogram WriteCSV should be a sink."

        testCase "quantiles over histogram data can feed image parameters" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let histogram =
                node "histogram" "HistogramData" []

            let quantiles =
                node "quantiles" "Quantiles"
                    [ p "histogram" "" true
                      p "q1" "0.5" false
                      p "useQ2" "true" false
                      p "q2" "0.01" false
                      p "useQ3" "false" false
                      p "q3" "0.99" false
                      p "useQ4" "false" false
                      p "q4" "0.25" false
                      p "useQ5" "false" false
                      p "q5" "0.75" false ]

            let shift =
                node "shift" "ShiftScale"
                    [ p "type" "Float64" false
                      p "shift" "0.0" true
                      p "scale" "1.0" false ]

            let code =
                graph
                    [ read; histogram; quantiles; shift ]
                    [ edge "read" "output" 0 "histogram" "input" 0
                      edge "histogram" "reducerOutput" 0 "quantiles" "parameterInput" 0
                      edge "quantiles" "reducerOutput" 1 "shift" "parameterInput" 1
                      edge "read" "output" 0 "shift" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Histogram0 =" "Linked histogram data should be bound before quantiles."
            Expect.stringContains code "let Quantiles0 = quantiles [0.5; 0.01] Histogram0" "Quantile binding should include q1 and enabled extra slots."
            Expect.stringContains code ">=> shiftScale<float> Quantiles0[1] 1.0" "Quantile output should be usable as a linked scalar parameter."

        testCase "histogram threshold estimators can feed standard threshold" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let histogram =
                node "histogram" "HistogramData" []

            let otsu =
                node "otsu" "OtsuThresholdFromHistogram"
                    [ p "histogram" "" true ]

            let threshold =
                node "threshold" "Threshold"
                    [ p "type" "Float64" false
                      p "lower" "0.0" true
                      p "upper" "infinity" false ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; histogram; otsu; threshold; write ]
                    [ edge "read" "output" 0 "histogram" "input" 0
                      edge "histogram" "reducerOutput" 0 "otsu" "parameterInput" 0
                      edge "otsu" "scalarOutput" 0 "threshold" "parameterInput" 1
                      edge "read" "output" 0 "threshold" "input" 0
                      edge "threshold" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Histogram0 =" "Linked histogram data should be bound before threshold estimation."
            Expect.stringContains code "let Float640 = otsuThresholdFromHistogram Histogram0" "Otsu threshold estimation should be a scalar binding."
            Expect.stringContains code ">=> threshold Float640 infinity" "The scalar threshold should feed the standard threshold stage."

        testCase "histogram equalization consumes linked histogram data" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "UInt16" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let histogram =
                node "histogram" "HistogramData" []

            let equalize =
                node "equalize" "HistogramEqualization"
                    [ p "type" "UInt16" false
                      p "histogram" "" true ]

            let write =
                node "write" "Write"
                    [ p "output" "out" false
                      p "suffix" ".tiff" false ]

            let code =
                graph
                    [ read; histogram; equalize; write ]
                    [ edge "read" "output" 0 "histogram" "input" 0
                      edge "histogram" "reducerOutput" 0 "equalize" "parameterInput" 1
                      edge "read" "output" 0 "equalize" "input" 0
                      edge "equalize" "output" 0 "write" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code "let Histogram0 =" "Linked histogram data should be bound before image equalization."
            Expect.stringContains code ">=> histogramEqualization<uint16> Histogram0" "Histogram equalization should consume the linked histogram map."

        testCase "intensity stretch lowers to StackProcessing stage" <| fun _ ->
            let read =
                node "read" "Read"
                    [ p "availableMemory" "1024" false
                      p "type" "Float64" false
                      p "input" "input" false
                      p "suffix" ".tiff" false ]

            let stretch =
                node "stretch" "IntensityStretch"
                    [ p "type" "Float64" false
                      p "inputMinimum" "10" false
                      p "inputMaximum" "20" false
                      p "outputMinimum" "0" false
                      p "outputMaximum" "1" false ]

            let code =
                graph
                    [ read; stretch ]
                    [ edge "read" "output" 0 "stretch" "input" 0 ]
                |> PipelineCodeGenerator.generateSavedGraph

            Expect.stringContains code ">=> intensityStretch<float> 10.0 20.0 0.0 1.0" "Intensity stretch should lower with typed linear range parameters."
    ]
