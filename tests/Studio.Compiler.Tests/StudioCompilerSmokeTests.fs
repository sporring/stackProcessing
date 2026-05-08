module Tests.StudioCompilerSmokeTests

open System
open System.Diagnostics
open System.IO
open System.Security
open System.Text
open Expecto
open Studio.Graph
open Studio.Compiler

type private Outlet =
    { Nodes: SavedNode list
      Edges: SavedEdge list
      NodeId: string
      Kind: string
      Port: int }

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

let private numericName numericType =
    NumericType.toString numericType

let private numericPortType portType =
    match portType with
    | PortType.Image NumericType.Number -> NumericType.Float64
    | PortType.Image numericType -> numericType
    | _ -> NumericType.Float64

let private savedNodeFromDefinition caseId (definition: Function) =
    let overrides =
        match definition.Id with
        | "Convolve" -> Map.ofList [ "kernel", "(Image<float>([1u; 1u]))" ]
        | "CreateByEuler2DTransform" -> Map.ofList [ "width", "16"; "height", "16"; "depth", "4"; "boxSize", "4" ]
        | "PermuteAxes" -> Map.ofList [ "axes", "(0u, 1u, 2u)" ]
        | _ -> Map.empty

    let parameters =
        definition.Parameters
        |> List.map (fun parameter ->
            let value =
                overrides
                |> Map.tryFind parameter.Key
                |> Option.defaultValue parameter.DefaultValue

            p parameter.Key value false)

    node $"target_{caseId}" definition.Id parameters

let private imageSource caseId numericType =
    let source =
        node $"source_{caseId}" "Zero"
            [ p "availableMemory" "1048576" false
              p "type" (numericName numericType) false
              p "width" "8" false
              p "height" "8" false
              p "depth" "4" false ]

    { Nodes = [ source ]
      Edges = []
      NodeId = source.Id
      Kind = "output"
      Port = 0 }

let rec private sourceFor caseId portType =
    match portType with
    | PortType.Image NumericType.Complex ->
        let real = imageSource $"{caseId}_real" NumericType.Float64
        let imag = imageSource $"{caseId}_imag" NumericType.Float64
        let combine = node $"source_{caseId}_complex" "ComplexFromReIm" []

        { Nodes = real.Nodes @ imag.Nodes @ [ combine ]
          Edges =
              real.Edges
              @ imag.Edges
              @ [ edge real.NodeId real.Kind real.Port combine.Id "input" 0
                  edge imag.NodeId imag.Kind imag.Port combine.Id "input" 1 ]
          NodeId = combine.Id
          Kind = "output"
          Port = 0 }
    | PortType.Image numericType ->
        imageSource caseId (if numericType = NumericType.Number then NumericType.Float64 else numericType)
    | PortType.Custom "VectorImageFloat64" ->
        let x = imageSource $"{caseId}_x" NumericType.Float64
        let y = imageSource $"{caseId}_y" NumericType.Float64
        let combine = node $"source_{caseId}_vector" "ToVectorImage" []

        { Nodes = x.Nodes @ y.Nodes @ [ combine ]
          Edges =
              x.Edges
              @ y.Edges
              @ [ edge x.NodeId x.Kind x.Port combine.Id "input" 0
                  edge y.NodeId y.Kind y.Port combine.Id "input" 1 ]
          NodeId = combine.Id
          Kind = "output"
          Port = 0 }
    | PortType.Custom "PointSet" ->
        let source =
            node $"source_{caseId}_points" "ReadPointSet"
                [ p "availableMemory" "1048576" false
                  p "input" "points.csv" false ]

        { Nodes = [ source ]; Edges = []; NodeId = source.Id; Kind = "output"; Port = 0 }
    | PortType.Custom "Mesh" ->
        let image = imageSource $"{caseId}_mesh_image" NumericType.Float64
        let mesh =
            node $"source_{caseId}_mesh" "MarchingCubes"
                [ p "type" "Float64" false
                  p "surfaceValue" "1.0" false ]

        { Nodes = image.Nodes @ [ mesh ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port mesh.Id "input" 0 ]
          NodeId = mesh.Id
          Kind = "output"
          Port = 0 }
    | PortType.Custom "Float64Matrix" ->
        let points = sourceFor $"{caseId}_matrix_points" (PortType.Custom "PointSet")
        let distances =
            node $"source_{caseId}_matrix" "PointPairDistances"
                [ p "xUnit" "1.0" false
                  p "yUnit" "1.0" false
                  p "zUnit" "1.0" false ]

        { Nodes = points.Nodes @ [ distances ]
          Edges = points.Edges @ [ edge points.NodeId points.Kind points.Port distances.Id "input" 0 ]
          NodeId = distances.Id
          Kind = "reducerOutput"
          Port = 0 }
    | PortType.Custom "StreamedObjects" ->
        let image = imageSource $"{caseId}_objects_image" NumericType.UInt8
        let objects =
            node $"source_{caseId}_objects" "StreamConnectedObjects"
                [ p "connectivity" "Six" false ]

        { Nodes = image.Nodes @ [ objects ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port objects.Id "input" 0 ]
          NodeId = objects.Id
          Kind = "output"
          Port = 0 }
    | PortType.Custom "TranslationTable" ->
        let labels = sourceFor $"{caseId}_labels" BuiltInCatalog.connectedComponentLabels
        let table = node $"source_{caseId}_table" "ComponentTranslationTable" [ p "windowSize" "3" false ]

        { Nodes = labels.Nodes @ [ table ]
          Edges = labels.Edges @ [ edge labels.NodeId labels.Kind labels.Port table.Id "input" 0 ]
          NodeId = table.Id
          Kind = "reducerOutput"
          Port = 0 }
    | PortType.Custom "BiasModel" ->
        let image = imageSource $"{caseId}_bias_image" NumericType.Float64
        let model =
            node $"source_{caseId}_bias" "FitBiasModel"
                [ p "type" "Float64" false
                  p "order" "2" false
                  p "depth" "4" false ]

        { Nodes = image.Nodes @ [ model ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port model.Id "input" 0 ]
          NodeId = model.Id
          Kind = "reducerOutput"
          Port = 0 }
    | PortType.Custom "SerialSliceManifest" ->
        let image = imageSource $"{caseId}_manifest_image" NumericType.Float64
        let manifest =
            node $"source_{caseId}_manifest" "SerialImageTranslationManifest"
                [ p "type" "Float64" false
                  p "maxShift" "4" false ]

        { Nodes = image.Nodes @ [ manifest ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port manifest.Id "input" 0 ]
          NodeId = manifest.Id
          Kind = "reducerOutput"
          Port = 0 }
    | PortType.Scalar BasicType.Map ->
        let image = imageSource $"{caseId}_histogram_image" NumericType.Float64
        let histogram = node $"source_{caseId}_histogram" "HistogramData" []

        { Nodes = image.Nodes @ [ histogram ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port histogram.Id "input" 0 ]
          NodeId = histogram.Id
          Kind = "reducerOutput"
          Port = 0 }
    | PortType.Custom "IntList"
    | PortType.Custom "UInt64List"
    | PortType.Scalar _ ->
        let source =
            node $"source_{caseId}_scalar" "Scalar"
                [ p "type" "Float64" false
                  p "value" "1.0" false ]

        { Nodes = [ source ]; Edges = []; NodeId = source.Id; Kind = "scalarOutput"; Port = 0 }
    | PortType.Tuple (PortType.Image NumericType.UInt64, PortType.Scalar (BasicType.Numeric NumericType.UInt64)) ->
        let image = imageSource $"{caseId}_cc_image" NumericType.UInt8
        let labels = node $"source_{caseId}_cc" "ConnectedComponents" [ p "windowSize" "3" false ]

        { Nodes = image.Nodes @ [ labels ]
          Edges = image.Edges @ [ edge image.NodeId image.Kind image.Port labels.Id "input" 0 ]
          NodeId = labels.Id
          Kind = "output"
          Port = 0 }
    | PortType.Any ->
        sourceFor caseId (PortType.Custom "PointSet")
    | _ ->
        imageSource caseId NumericType.Float64

let private outputKindFor functionId portType =
    match functionId with
    | "Scalar"
    | "FileDirectory"
    | "ScalarOp"
    | "ScalarFunction"
    | "OtsuThresholdFromHistogram"
    | "MomentsThresholdFromHistogram" -> "scalarOutput"
    | "ComputeStats"
    | "SurfaceArea"
    | "Volume"
    | "PointPairDistances"
    | "AffineRegistration"
    | "FitBiasModel"
    | "FitBiasModelMasked"
    | "SerialKeypointTranslationManifest"
    | "SerialImageTranslationManifest"
    | "GetStackInfo"
    | "GetChunkInfo"
    | "GetZarrInfo"
    | "GetNexusInfo"
    | "ComponentTranslationTable"
    | "HistogramData"
    | "EstimateHistogram"
    | "Quantiles" -> "reducerOutput"
    | _ ->
        match portType with
        | PortType.Scalar _
        | PortType.Custom "TranslationTable" -> "reducerOutput"
        | _ -> "output"

let private sinkFor caseId functionId portType =
    let targetKind = outputKindFor functionId portType

    match portType with
    | PortType.Image NumericType.Complex ->
        [ node $"sink_{caseId}_modulus" "ComplexModulus" []
          node $"sink_{caseId}_write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_modulus" "input" 0
          edge $"sink_{caseId}_modulus" "output" 0 $"sink_{caseId}_write" "input" 0 ]
    | PortType.Image _ ->
        [ node $"sink_{caseId}_write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_write" "input" 0 ]
    | PortType.Custom "VectorImageFloat64" ->
        [ node $"sink_{caseId}_element" "VectorElement" [ p "component" "0" false ]
          node $"sink_{caseId}_write" "Write" [ p "output" "out" false; p "suffix" ".tiff" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_element" "input" 0
          edge $"sink_{caseId}_element" "output" 0 $"sink_{caseId}_write" "input" 0 ]
    | PortType.Custom "PointSet" ->
        [ node $"sink_{caseId}_points" "WritePointSet" [ p "output" "points" false; p "suffix" ".csv" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_points" "input" 0 ]
    | PortType.Custom "Mesh" ->
        [ node $"sink_{caseId}_mesh" "WriteMesh" [ p "output" "mesh.obj" false; p "format" "auto" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_mesh" "input" 0 ]
    | PortType.Custom "Float64Matrix" ->
        [ node $"sink_{caseId}_matrix" "WriteMatrix" [ p "output" "matrix" false; p "suffix" ".csv" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_matrix" "input" 0 ]
    | PortType.Custom "StreamedObjects" ->
        [ node $"sink_{caseId}_paint" "PaintObjectsCropped" []
          node $"sink_{caseId}_write" "Write" [ p "output" "objects" false; p "suffix" ".tiff" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_paint" "input" 0
          edge $"sink_{caseId}_paint" "output" 0 $"sink_{caseId}_write" "input" 0 ]
    | PortType.Tuple _ ->
        [ node $"sink_{caseId}_table" "ComponentTranslationTable" [ p "windowSize" "3" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_table" "input" 0 ]
    | PortType.Scalar _
    | PortType.Custom _ ->
        [ node $"sink_{caseId}_print" "Print" [ p "format" "{input1}" false; p "input1" "" true ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_print" "parameterInput" 1 ]
    | PortType.Any ->
        [ node $"sink_{caseId}_csv" "WriteCSV" [ p "output" "any" false; p "dataKind" "PointSet" false ] ],
        [ edge $"target_{caseId}" targetKind 0 $"sink_{caseId}_csv" "input" 0 ]
    | PortType.Unit ->
        [], []

let private parameterSourceFor caseId parameterKey =
    match parameterKey with
    | "model" -> sourceFor $"{caseId}_param_model" (PortType.Custom "BiasModel")
    | "manifest" -> sourceFor $"{caseId}_param_manifest" (PortType.Custom "SerialSliceManifest")
    | "translationTable" -> sourceFor $"{caseId}_param_table" (PortType.Custom "TranslationTable")
    | "histogram" -> sourceFor $"{caseId}_param_histogram" (PortType.Scalar BasicType.Map)
    | "input" -> sourceFor $"{caseId}_param_input" (PortType.Scalar BasicType.Map)
    | _ -> sourceFor $"{caseId}_param_scalar" (PortType.Scalar(BasicType.Numeric NumericType.Float64))

let private effectiveInputPortType (definition: Function) (port: Port) =
    match port.Type with
    | PortType.Image NumericType.Number ->
        definition.Parameters
        |> List.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.DefaultValue)
        |> Option.map PortType.Image
        |> Option.defaultValue port.Type
    | _ ->
        port.Type

let private graphForDefinition caseIndex (definition: Function) =
    let caseId = $"{caseIndex}_{definition.Id}"
    let target = savedNodeFromDefinition caseId definition

    let inputOutlets =
        definition.Inputs
        |> List.mapi (fun index input -> index, sourceFor $"{caseId}_input{index}" (effectiveInputPortType definition input))

    let inputNodes = inputOutlets |> List.collect (fun (_, outlet) -> outlet.Nodes)
    let inputEdges =
        inputOutlets
        |> List.collect (fun (index, outlet) -> outlet.Edges @ [ edge outlet.NodeId outlet.Kind outlet.Port target.Id "input" index ])

    let linkedParameterKeys =
        match definition.Id with
        | "CorrectBias"
        | "CorrectBiasMasked" -> [ "model" ]
        | "SerialApplyManifest"
        | "SerialApplyManifestInBoundingBox" -> [ "manifest" ]
        | "CollapseComponentLabels" -> [ "translationTable" ]
        | "HistogramEqualization"
        | "Quantiles"
        | "OtsuThresholdFromHistogram"
        | "MomentsThresholdFromHistogram" -> [ "histogram" ]
        | "Chart" -> [ "input" ]
        | _ -> []

    let parameterOutlets =
        linkedParameterKeys
        |> List.choose (fun key ->
            definition.Parameters
            |> List.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> key, index, parameterSourceFor caseId key))

    let parameterNodes = parameterOutlets |> List.collect (fun (_, _, outlet) -> outlet.Nodes)
    let parameterEdges =
        parameterOutlets
        |> List.collect (fun (_, index, outlet) -> outlet.Edges @ [ edge outlet.NodeId outlet.Kind outlet.Port target.Id "parameterInput" index ])

    let targetWithLinkedParameters =
        if linkedParameterKeys.IsEmpty then
            target
        else
            let parameters =
                target.Parameters
                |> Array.map (fun parameter ->
                    if linkedParameterKeys |> List.contains parameter.Key then
                        { parameter with UseInput = true; Value = "" }
                    else
                        parameter)

            { target with Parameters = parameters }

    let outputNodes, outputEdges =
        match definition.Outputs with
        | first :: _ -> sinkFor caseId definition.Id first.Type
        | [] -> [], []

    graph
        (inputNodes @ parameterNodes @ [ targetWithLinkedParameters ] @ outputNodes)
        (inputEdges @ parameterEdges @ outputEdges)

let private stripOpens (program: string) =
    program.Split([| Environment.NewLine |], StringSplitOptions.None)
    |> Array.filter (fun line -> not (line.StartsWith("open ", StringComparison.Ordinal)))
    |> String.concat Environment.NewLine

let private indent spaces (text: string) =
    let prefix = String.replicate spaces " "

    text.Split([| Environment.NewLine |], StringSplitOptions.None)
    |> Array.map (fun line -> if String.IsNullOrWhiteSpace line then line else prefix + line)
    |> String.concat Environment.NewLine

let private writeSmokeProject directory (source: string) =
    Directory.CreateDirectory(directory) |> ignore

    let repositoryRoot = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "..", ".."))
    let projectPath = Path.Combine(directory, "StudioCompilerSmoke.fsproj")
    let programPath = Path.Combine(directory, "Program.fs")
    let stackProcessingProject = Path.Combine(repositoryRoot, "src", "StackProcessing", "StackProcessing.fsproj")
    let stackProcessingCoreProject = Path.Combine(repositoryRoot, "src", "StackProcessing.Core", "StackProcessing.Core.fsproj")
    let simpleItkManaged = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpManaged.dll")
    let simpleItkWindowsNative = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpNative.dll")
    let simpleItkLinuxNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.so")
    let simpleItkMacNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.dylib")

    let xml value = SecurityElement.Escape(value)

    let projectFile =
        $"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <Reference Include="SimpleITKCSharp">
      <HintPath>{xml simpleItkManaged}</HintPath>
      <Private>true</Private>
    </Reference>
    <ProjectReference Include="{xml stackProcessingProject}" />
    <ProjectReference Include="{xml stackProcessingCoreProject}" />
    <PackageReference Include="Plotly.NET" Version="5.1.0" />
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('Windows'))">
    <None Include="{xml simpleItkWindowsNative}">
      <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
      <TargetPath>libSimpleITKCSharpNative.dll</TargetPath>
    </None>
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('Linux'))">
    <None Include="{xml simpleItkLinuxNative}" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
  <ItemGroup Condition="$([MSBuild]::IsOSPlatform('OSX'))">
    <None Include="{xml simpleItkMacNative}" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
  <ItemGroup>
    <Compile Include="Program.fs" />
  </ItemGroup>
</Project>
"""

    File.WriteAllText(projectPath, projectFile, Encoding.UTF8)
    File.WriteAllText(programPath, source, Encoding.UTF8)
    projectPath

let private runDotnetBuild (projectPath: string) =
    let dotnet =
        let macInstall = "/usr/local/share/dotnet/dotnet"
        if File.Exists macInstall then macInstall else "dotnet"

    let info = ProcessStartInfo(dotnet, $"build \"{projectPath}\" --nologo /nr:false")
    info.RedirectStandardOutput <- true
    info.RedirectStandardError <- true
    info.UseShellExecute <- false
    info.WorkingDirectory <- Path.GetDirectoryName(projectPath)

    use proc = Process.Start(info)
    let output = proc.StandardOutput.ReadToEndAsync()
    let error = proc.StandardError.ReadToEndAsync()

    if proc.WaitForExit(TimeSpan.FromMinutes(2.0)) then
        proc.ExitCode, output.Result + error.Result
    else
        try
            proc.Kill(entireProcessTree = true)
        with _ ->
            ()

        -1, output.Result + error.Result + Environment.NewLine + "dotnet build timed out."

let private generatedProgramFor caseName savedGraph =
    let roundTripped =
        savedGraph
        |> PipelineGraphStorage.serialize
        |> PipelineGraphStorage.deserialize

    let body =
        roundTripped
        |> PipelineCodeGenerator.generateSavedGraph
        |> stripOpens

    $"let {caseName} () ={Environment.NewLine}{indent 4 body}"

let private compileSmokeCases cases =
    let builder = StringBuilder()
    builder.AppendLine("module StudioCompilerSmoke") |> ignore
    builder.AppendLine("open StackProcessing") |> ignore
    builder.AppendLine("open Plotly.NET") |> ignore
    builder.AppendLine() |> ignore

    cases
    |> List.iter (fun (name, savedGraph) ->
        builder.AppendLine(generatedProgramFor name savedGraph) |> ignore
        builder.AppendLine() |> ignore)

    builder.AppendLine("[<EntryPoint>]") |> ignore
    builder.AppendLine("let main _ = 0") |> ignore

    let directory =
        Path.Combine("/private/tmp", "StackProcessingStudioCompilerSmoke", string Environment.ProcessId)

    let projectPath = writeSmokeProject directory (builder.ToString())
    runDotnetBuild projectPath

[<Tests>]
let smokeSuite =
    testList "Studio.Compiler generated F# smoke tests" [
        testCase "catalog box graphs compile as F#" <| fun _ ->
            let cases =
                BuiltInCatalog.orderedFunctions
                |> List.mapi (fun index definition ->
                    let name =
                        definition.Id
                        |> Seq.map (fun c -> if Char.IsLetterOrDigit c then c else '_')
                        |> Seq.toArray
                        |> fun chars -> new System.String(chars)
                        |> fun id -> $"case_{index}_{id}"

                    name, graphForDefinition index definition)

            let exitCode, output = compileSmokeCases cases

            Expect.equal exitCode 0 $"Generated F# programs should build. dotnet build output:{Environment.NewLine}{output}"
    ]
