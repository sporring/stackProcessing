namespace Studio.ViewModels

open System
open System.Collections.Generic
open System.Collections.ObjectModel
open System.Collections.Specialized
open System.ComponentModel
open System.Diagnostics
open System.Globalization
open System.IO
open System.Runtime.CompilerServices
open System.Security
open System.Text
open System.Text.RegularExpressions
open System.Threading
open System.Windows.Input
open Avalonia.Threading
open Studio.Compiler
open Studio.Graph
open NodeEditor.Mvvm
open NodeEditor.Model
open Studio.Models
open Studio.Services

type private SimpleCommand(execute: obj -> unit, canExecute: obj -> bool) =
    let canExecuteChanged = Event<EventHandler, EventArgs>()

    interface ICommand with
        member _.CanExecute(parameter) = canExecute parameter
        member _.Execute(parameter) = execute parameter

        [<CLIEvent>]
        member _.CanExecuteChanged = canExecuteChanged.Publish

type PipelinePinKind =
    | DataInput
    | DataOutput
    | ParameterInput
    | ScalarOutput
    | ReducerOutput

module PipelinePinKind =
    let toString kind =
        match kind with
        | DataInput -> "dataInput"
        | DataOutput -> "dataOutput"
        | ParameterInput -> "parameterInput"
        | ScalarOutput -> "scalarOutput"
        | ReducerOutput -> "reducerOutput"

    let ofString value =
        match value with
        | "dataInput" -> DataInput
        | "parameterInput" -> ParameterInput
        | "scalarOutput" -> ScalarOutput
        | "reducerOutput" -> ReducerOutput
        | "dataOutput" -> DataOutput
        | _ -> invalidOp $"Unknown pipeline pin kind: {value}"

    let isInput kind =
        match kind with
        | DataInput
        | ParameterInput -> true
        | DataOutput
        | ScalarOutput
        | ReducerOutput -> false

    let isOutput kind = not (isInput kind)

module private PipelinePinGeometry =
    let size = 14.
    let halfSize = size / 2.
    let triangleOffset = 0.

type PipelinePinViewModel(alignment: PinAlignment, port: Port, kind: PipelinePinKind, ?parameterKey: string) =
    inherit PinViewModel()

    let mutable isActive = kind <> ParameterInput
    let mutable portValue = port

    member this.Port
        with get () = portValue
        and set value =
            if portValue <> value then
                portValue <- value
                this.OnPropertyChanged(nameof this.Port)

    member _.Kind = kind
    member _.ParameterKey = defaultArg parameterKey ""
    member _.IsActive = isActive

    member _.PinOpacity =
        if kind = ParameterInput && not isActive then 0.0 else 1.0

    member this.SetActive(value: bool) =
        if isActive <> value then
            isActive <- value
            this.OnPropertyChanged(nameof this.IsActive)
            this.OnPropertyChanged(nameof this.PinOpacity)

    member _.PinBrush =
        match kind with
        | ParameterInput
        | ScalarOutput
        | ReducerOutput -> "#8A5A22"
        | DataInput
        | DataOutput -> "#FFFFFF"

    member _.IsInput = PipelinePinKind.isInput kind
    member _.IsOutput = PipelinePinKind.isOutput kind

type ArrangeSettingsViewModel() =
    inherit ViewModelBase()

    let defaultBoundingBoxCompact = 0.005
    let defaultBoundingBoxSquare = 0.
    let defaultBoxSpacing = 5.
    let defaultBoxSpacingSharpness = 1.
    let defaultConnectorDirection = 1.
    let defaultConnectorLength = 0.01
    let defaultConnectorAvoidsBoxes = 1.
    let defaultConnectorCrossings = 0.
    let defaultStayNearUserLayout = 0.
    let defaultDamping = 0.75
    let defaultVisualizationDelayMs = 28.0

    let mutable boundingBoxCompact = defaultBoundingBoxCompact
    let mutable boundingBoxSquare = defaultBoundingBoxSquare
    let mutable boxSpacing = defaultBoxSpacing
    let mutable boxSpacingSharpness = defaultBoxSpacingSharpness
    let mutable connectorDirection = defaultConnectorDirection
    let mutable connectorLength = defaultConnectorLength
    let mutable connectorAvoidsBoxes = defaultConnectorAvoidsBoxes
    let mutable connectorCrossings = defaultConnectorCrossings
    let mutable stayNearUserLayout = defaultStayNearUserLayout
    let mutable damping = defaultDamping
    let mutable visualizationDelayMs = defaultVisualizationDelayMs

    member this.BoundingBoxCompact
        with get () = boundingBoxCompact
        and set value = this.SetProperty(&boundingBoxCompact, value) |> ignore

    member this.BoundingBoxSquare
        with get () = boundingBoxSquare
        and set value = this.SetProperty(&boundingBoxSquare, value) |> ignore

    member this.BoxSpacing
        with get () = boxSpacing
        and set value = this.SetProperty(&boxSpacing, value) |> ignore

    member this.BoxSpacingSharpness
        with get () = boxSpacingSharpness
        and set value = this.SetProperty(&boxSpacingSharpness, value) |> ignore

    member this.ConnectorDirection
        with get () = connectorDirection
        and set value = this.SetProperty(&connectorDirection, value) |> ignore

    member this.ConnectorLength
        with get () = connectorLength
        and set value = this.SetProperty(&connectorLength, value) |> ignore

    member this.ConnectorAvoidsBoxes
        with get () = connectorAvoidsBoxes
        and set value = this.SetProperty(&connectorAvoidsBoxes, value) |> ignore

    member this.ConnectorCrossings
        with get () = connectorCrossings
        and set value = this.SetProperty(&connectorCrossings, value) |> ignore

    member this.StayNearUserLayout
        with get () = stayNearUserLayout
        and set value = this.SetProperty(&stayNearUserLayout, value) |> ignore

    member this.Damping
        with get () = damping
        and set value = this.SetProperty(&damping, value) |> ignore

    member this.VisualizationDelayMs
        with get () = visualizationDelayMs
        and set value = this.SetProperty(&visualizationDelayMs, value) |> ignore

    member this.ResetDefaults() =
        this.BoundingBoxCompact <- defaultBoundingBoxCompact
        this.BoundingBoxSquare <- defaultBoundingBoxSquare
        this.BoxSpacing <- defaultBoxSpacing
        this.BoxSpacingSharpness <- defaultBoxSpacingSharpness
        this.ConnectorDirection <- defaultConnectorDirection
        this.ConnectorLength <- defaultConnectorLength
        this.ConnectorAvoidsBoxes <- defaultConnectorAvoidsBoxes
        this.ConnectorCrossings <- defaultConnectorCrossings
        this.StayNearUserLayout <- defaultStayNearUserLayout
        this.Damping <- defaultDamping
        this.VisualizationDelayMs <- defaultVisualizationDelayMs

module private PortMapping =
    let private basicTypeLabel basicType =
        match basicType with
        | BasicType.String -> "string"
        | BasicType.Bool -> "bool"
        | BasicType.Map -> "Map"
        | BasicType.Unit -> "unit"
        | BasicType.Numeric numericType -> NumericType.toString numericType

    let customParameterPort key label =
        { Name = $"{key}: {label}"
          Type = Custom label }

    let parameterPort (parameter: PipelineParameterViewModel) =
        { Name = $"{parameter.Key}: {basicTypeLabel parameter.ParameterType}"
          Type = Scalar parameter.ParameterType }

    let anyParameterPort key =
        { Name = $"{key}: Any"
          Type = Any }

    let portTypeLabel portType =
        match portType with
        | Scalar basicType -> BasicType.toString basicType
        | Image numericType -> NumericType.toString numericType
        | Custom label -> label
        | Tuple _ -> "Tuple"
        | Any -> "Any"
        | Unit -> "Unit"

    let streamValueName portType =
        $"I: {portTypeLabel portType}"

module private NodeTitle =
    let quotedString (value: string) =
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

module private ScalarNode =
    let typeOptions =
        [ Numeric UInt8
          Numeric Int8
          Numeric UInt16
          Numeric Int16
          Numeric UInt32
          Numeric Int32
          Numeric UInt64
          Numeric Int64
          Numeric Float32
          Numeric Float64
          Numeric Complex
          Bool
          String ]
        |> List.map BasicType.toString

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> BasicType.tryParse parameter.Value)
        |> Option.defaultValue (Numeric Float64)

    let outputPort (state: PipelineNodeState) =
        { Name = "Value"
          Type = Scalar(selectedType state) }

    let defaultValue scalarType =
        match scalarType with
        | BasicType.Bool -> "true"
        | BasicType.String -> "value"
        | BasicType.Map -> "map"
        | BasicType.Unit -> "()"
        | BasicType.Numeric numericType ->
            match numericType with
            | Float32
            | Float64
            | Number
            | Complex -> "1.0"
            | UInt8
            | Int8
            | UInt16
            | Int16
            | UInt32
            | Int32
            | UInt64
            | Int64 -> "1"

    let private isInteger (value: string) =
        let mutable parsed = 0L
        Int64.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, &parsed)

    let private isUnsignedInteger (value: string) =
        let mutable parsed = 0UL
        UInt64.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, &parsed)

    let private isFloat (value: string) =
        let mutable parsed = 0.0
        Double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, &parsed)

    let private isStandardNumericName (value: string) =
        String.Equals(value, "e", StringComparison.OrdinalIgnoreCase)
        || String.Equals(value, "pi", StringComparison.OrdinalIgnoreCase)

    let isValidValue scalarType (value: string) =
        let trimmed = value.Trim()

        match scalarType with
        | BasicType.Bool ->
            String.Equals(trimmed, "true", StringComparison.OrdinalIgnoreCase)
            || String.Equals(trimmed, "false", StringComparison.OrdinalIgnoreCase)
        | BasicType.String ->
            true
        | BasicType.Map ->
            not (String.IsNullOrWhiteSpace trimmed)
        | BasicType.Unit ->
            trimmed = "()"
        | BasicType.Numeric numericType ->
            match numericType with
            | UInt8
            | UInt16
            | UInt32
            | UInt64 -> isUnsignedInteger trimmed || isStandardNumericName trimmed
            | Int8
            | Int16
            | Int32
            | Int64 -> isInteger trimmed || isStandardNumericName trimmed
            | Float32
            | Float64
            | Number
            | Complex -> isFloat trimmed || isStandardNumericName trimmed

    let ensureValueMatchesType (state: PipelineNodeState) =
        let selectedType = selectedType state

        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "value")
        |> Option.iter (fun parameter ->
            if not (isValidValue selectedType parameter.Value) then
                parameter.Value <- defaultValue selectedType)

    let title (state: PipelineNodeState) =
        let value =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "value")
            |> Option.map (fun parameter -> parameter.Value.Trim())
            |> Option.filter (String.IsNullOrWhiteSpace >> not)

        match selectedType state, value with
        | BasicType.String, Some value -> NodeTitle.quotedString value
        | _, Some value -> value
        | _ -> state.Definition.DisplayName

module private FileDirectoryNode =
    let kindOptions = [ "File"; "Directory" ]

    let title (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "value")
        |> Option.map (fun parameter -> parameter.Value.Trim())
        |> Option.filter (String.IsNullOrWhiteSpace >> not)
        |> Option.map NodeTitle.quotedString
        |> Option.defaultValue state.Definition.DisplayName

module private StandardFunctionOptions =
    let values =
        [ "abs"; "acos"; "asin"; "atan"; "cos"; "sin"; "tan"; "exp"; "log10"; "log"; "round"; "sqrt"; "square" ]

module private ScalarOpNode =
    let typeOptions =
        [ Numeric UInt8
          Numeric Int8
          Numeric UInt16
          Numeric Int16
          Numeric UInt32
          Numeric Int32
          Numeric UInt64
          Numeric Int64
          Numeric Float32
          Numeric Float64
          Numeric Complex ]
        |> List.map BasicType.toString

    let operationOptions = [ "+"; "-"; "*"; "/" ]

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> BasicType.tryParse parameter.Value)
        |> Option.defaultValue (Numeric Float64)

    let scalarPort (label: string) (state: PipelineNodeState) =
        let selectedType = selectedType state
        let label = label.ToLowerInvariant()

        { Name = $"{label}: {BasicType.toString selectedType}"
          Type = Scalar selectedType }

    let outputPort (state: PipelineNodeState) =
        let selectedType = selectedType state

        { Name = BasicType.toString selectedType
          Type = Scalar selectedType }

    let title (state: PipelineNodeState) =
        let operation =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "operation")
            |> Option.map _.Value
            |> Option.filter (fun value -> operationOptions |> List.contains value)
            |> Option.defaultValue "*"

        $"a {operation} b"

module private ScalarFunctionNode =
    let functionOptions = StandardFunctionOptions.values

    let scalarPort =
        { Name = "a: Float64"
          Type = Scalar(BasicType.Numeric Float64) }

    let outputPort =
        { Name = "Float64"
          Type = Scalar(BasicType.Numeric Float64) }

    let title (state: PipelineNodeState) =
        let functionName =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "function")
            |> Option.map _.Value
            |> Option.filter (fun value -> functionOptions |> List.contains value)
            |> Option.defaultValue "sqrt"

        $"{functionName}(a)"

module private SourceImageNode =
    let hasInputTitle functionId =
        functionId = "Read" || functionId = "ReadRandom" || functionId = "EstimateHistogram" || functionId = "ReadRange" || functionId = "ReadSlab"

    let hasOutputTitle functionId =
        functionId = "Write" || functionId = "WriteThrough" || functionId = "WriteChunks"

    let hasFormatParameter functionId =
        hasInputTitle functionId
        || hasOutputTitle functionId
        || functionId = "WriteSlabSlices"
        || functionId = "GetChunkInfo"
        || functionId = "ResampleAffineTrilinearSlices"

    let typeOptions =
        [ UInt8
          Int8
          UInt16
          Int16
          UInt32
          Int32
          UInt64
          Int64
          Float32
          Float64
          Complex ]
        |> List.map NumericType.toString

    let suffixOptions =
        ImageFileFormat.formats
        |> List.map (fun format -> format.Label, format.Suffix)

    let readSuffixOptions =
        ImageFileFormat.readFormats
        |> List.map (fun format -> format.Label, format.Suffix)

    let readFormatOptions =
        [ "Image stack"
          "Volume file"
          "OME-Zarr"
          "NeXus/HDF5" ]

    let readSlabFormatOptions =
        [ "Chunked stack"
          "OME-Zarr"
          "NeXus/HDF5" ]

    let writeFormatOptions =
        [ "Image stack"
          "Volume file"
          "OME-Zarr"
          "NeXus/HDF5" ]

    let private readSourceFunctionIds =
        Set.ofList [ "Read"; "ReadRandom"; "EstimateHistogram"; "ReadRange"; "ReadSlab" ]

    let isReadSource functionId =
        readSourceFunctionIds |> Set.contains functionId

    let private syntheticSourceFunctionIds =
        Set.ofList [ "Zero"; "NormalNoise"; "SaltAndPepperNoise"; "ShotNoise"; "SpeckleNoise"; "CreateByEuler2DTransform" ]

    let isSyntheticSource functionId =
        syntheticSourceFunctionIds |> Set.contains functionId

    let selectedFormat (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "format")
        |> Option.map _.Value
        |> Option.defaultValue "Image stack"

    let private readVisibleParameterKeys (state: PipelineNodeState) =
        let common = Set.ofList [ "availableMemory"; "type"; "format"; "input" ]

        let zarr =
            Set.ofList [ "slabDepth"; "multiscaleIndex"; "datasetIndex"; "timepoint"; "channel"; "maxParallelChunks" ]

        let nexus =
            Set.ofList [ "datasetPath"; "slabDepth"; "frameAxis"; "yAxis"; "xAxis" ]

        match state.Definition.Id, selectedFormat state with
        | "ReadRandom", "Image stack" ->
            common |> Set.add "depth" |> Set.add "suffix"
        | "ReadRandom", "Volume file" ->
            common |> Set.add "depth" |> Set.add "suffix"
        | "Read", "Image stack"
        | "ReadSlab", "Chunked stack" ->
            common |> Set.add "suffix"
        | "Read", "Volume file" ->
            common |> Set.add "suffix"
        | "ReadRandom", "OME-Zarr" ->
            Set.union common zarr |> Set.add "depth" |> Set.remove "slabDepth"
        | "ReadRandom", "NeXus/HDF5" ->
            Set.union common nexus |> Set.add "depth" |> Set.remove "slabDepth"
        | ("Read" | "ReadSlab"), "OME-Zarr" ->
            Set.union common zarr
        | ("Read" | "ReadSlab"), "NeXus/HDF5" ->
            Set.union common nexus
        | _ ->
            state.Parameters
            |> Seq.map _.Key
            |> Set.ofSeq

    let private writeVisibleParameterKeys (state: PipelineNodeState) =
        let common = Set.ofList [ "format"; "output" ]
        let chunk = Set.ofList [ "depth"; "chunkX"; "chunkY"; "chunkZ" ]

        match selectedFormat state with
        | "Image stack" ->
            common |> Set.add "suffix"
        | "Volume file" ->
            common |> Set.add "suffix"
        | "OME-Zarr" ->
            Set.union common chunk
            |> Set.add "name"
            |> Set.add "physicalSizeX"
            |> Set.add "physicalSizeY"
            |> Set.add "physicalSizeZ"
            |> Set.add "maxConcurrentWrites"
        | "NeXus/HDF5" ->
            Set.union common chunk
            |> Set.add "datasetPath"
            |> Set.add "frameAxis"
            |> Set.add "yAxis"
            |> Set.add "xAxis"
        | _ ->
            state.Parameters
            |> Seq.map _.Key
            |> Set.ofSeq

    let parameterIsVisible (state: PipelineNodeState) key =
        if state.Definition.Id = "Read" || state.Definition.Id = "ReadRandom" || state.Definition.Id = "ReadSlab" then
            readVisibleParameterKeys state |> Set.contains key
        elif state.Definition.Id = "Write" then
            writeVisibleParameterKeys state |> Set.contains key
        else
            true

    let updateParameterVisibility (state: PipelineNodeState) =
        for parameter in state.Parameters do
            parameter.IsVisible <- parameterIsVisible state parameter.Key

    let private suffixOptionsForState (state: PipelineNodeState) =
        match state.Definition.Id, selectedFormat state with
        | "Read", "Volume file" ->
            readSuffixOptions
            |> List.filter (fun (_, value) -> value = ".tiff")
        | "Write", "Volume file" ->
            suffixOptions
            |> List.filter (fun (_, value) -> value = ".tiff")
        | functionId, _ when hasInputTitle functionId ->
            readSuffixOptions
        | _ ->
            suffixOptions

    let updateReadSuffixOptionStates (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "suffix")
        |> Option.iter (fun parameter ->
            let desired = suffixOptionsForState state
            let current = parameter.Options |> Seq.map (fun option -> option.Label, option.Value) |> Seq.toList
            let mutable optionsChanged = false

            if current <> desired then
                parameter.Options.Clear()
                optionsChanged <- true

                for label, value in desired do
                    parameter.Options.Add(ParameterOptionViewModel(label, value, true))
            else
                for option in parameter.Options do
                    option.IsEnabled <- true

            if parameter.Options |> Seq.exists (fun option -> option.Value = parameter.Value && option.IsEnabled) |> not then
                parameter.Options
                |> Seq.tryFind _.IsEnabled
                |> Option.iter (fun option -> parameter.Value <- option.Value)

            if optionsChanged then
                parameter.RefreshSelectedOption())

    let suffixOptionsFor functionId =
        if hasInputTitle functionId || functionId = "GetChunkInfo" || functionId = "ResampleAffineTrilinearSlices" then
            readSuffixOptions
        else
            suffixOptions

    let selectedSuffix (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "suffix")
        |> Option.map _.Value
        |> Option.defaultValue ".tiff"

    let supportedTypes (state: PipelineNodeState) =
        match state.Definition.Id with
        | functionId when isSyntheticSource functionId ->
            typeOptions
            |> List.choose NumericType.tryParse
        | functionId when isReadSource functionId ->
            typeOptions
            |> List.choose NumericType.tryParse
        | "Write" ->
            match selectedFormat state with
            | "OME-Zarr" -> [ UInt8; UInt16 ]
            | "NeXus/HDF5" -> [ UInt8; Int8; UInt16; Int16; UInt32; Int32; Float32; Float64 ]
            | "Volume file" -> ImageFileFormat.readSupportedTypes ".tiff"
            | _ -> selectedSuffix state |> ImageFileFormat.supportedTypes
        | _ when hasInputTitle state.Definition.Id ->
            selectedSuffix state
            |> ImageFileFormat.readSupportedTypes
        | _ ->
            selectedSuffix state
            |> ImageFileFormat.supportedTypes

    let supportedTypeOptions (state: PipelineNodeState) =
        supportedTypes state
        |> List.map NumericType.toString

    let selectedType (state: PipelineNodeState) =
        let supportedTypes = supportedTypes state

        let selected =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
            |> Option.defaultValue Float64

        if supportedTypes |> List.contains selected then
            selected
        else
            supportedTypes |> List.tryHead |> Option.defaultValue Float64

    let outputPort (state: PipelineNodeState) =
        let selectedType = selectedType state

        { Name = NumericType.toString selectedType
          Type = PortType.numericToImage selectedType }

    let writeInputPort (state: PipelineNodeState) =
        match supportedTypes state with
        | [ numericType ] ->
            { Name = NumericType.toString numericType
              Type = PortType.numericToImage numericType }
        | _ ->
            { Name = "Number"
              Type = PortType.Image Number }

    let supportsOutputType (state: PipelineNodeState) numericType =
        supportedTypes state
        |> List.contains numericType

    let private parameterTitle parameterKey fallback (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = parameterKey)
        |> Option.map (fun parameter ->
            if parameter.UseInput then
                parameter.Key
            else
                parameter.Value.Trim() |> NodeTitle.quotedString)
        |> Option.filter (String.IsNullOrWhiteSpace >> not)
        |> Option.defaultValue (NodeTitle.quotedString fallback)

    let private stripVolumeFileSuffix (state: PipelineNodeState) (value: string) =
        let trimmed = value.Trim()

        if selectedFormat state = "Volume file" then
            [ ".tiff"; ".tif" ]
            |> List.tryFind (fun suffix -> trimmed.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun suffix -> trimmed.Substring(0, trimmed.Length - suffix.Length))
            |> Option.defaultValue trimmed
        else
            trimmed

    let title (state: PipelineNodeState) =
        let parameterText =
            match state.Parameters |> Seq.tryFind (fun parameter -> parameter.Key = "input") with
            | Some parameter ->
                if parameter.UseInput then
                    parameter.Key
                else
                    parameter.Value |> stripVolumeFileSuffix state |> NodeTitle.quotedString
            | None -> parameterTitle "output" "output" state

        $"{state.Definition.DisplayName} {parameterText}"

module private PairOperationNode =
    let inputLabels = [| "I"; "J"; "K"; "L" |]

    let inputName index typeName =
        let label =
            inputLabels
            |> Array.tryItem index
            |> Option.defaultValue $"I{index + 1}"

        $"{label}: {typeName}"

    let typeOptions =
        [ UInt8
          Int8
          UInt16
          Int16
          UInt32
          Int32
          UInt64
          Int64
          Float32
          Float64
          Complex ]
        |> List.map NumericType.toString

    let operationOptions = [ "+"; "-"; "*"; "/"; "max"; "min" ]

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
        |> Option.defaultValue Float64

    let title (state: PipelineNodeState) =
        let operation =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "operation")
            |> Option.map _.Value
            |> Option.filter (fun value -> operationOptions |> List.contains value)
            |> Option.defaultValue "*"

        match state.Definition.Id, operation with
        | "ImageOpImage", "max" -> "max(I, J)"
        | "ImageOpImage", "min" -> "min(I, J)"
        | "ImageOpImage", _ -> $"I .{operation} J"
        | _ -> state.Definition.DisplayName

    let ports (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let portType = PortType.numericToImage selectedType

        [ { Name = inputName 0 typeName
            Type = portType }
          { Name = inputName 1 typeName
            Type = portType } ],
        [ { Name = typeName
            Type = portType } ]

module private CastNode =
    let typeOptions = SourceImageNode.typeOptions

    let private selectedParameterType key (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = key)
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
        |> Option.defaultValue Float64

    let ports (state: PipelineNodeState) =
        let sourceType = selectedParameterType "sourceType" state
        let sourceTypeName = NumericType.toString sourceType
        let targetType = selectedParameterType "targetType" state
        let targetTypeName = NumericType.toString targetType

        [ { Name = sourceTypeName
            Type = PortType.numericToImage sourceType } ],
        [ { Name = targetTypeName
            Type = PortType.numericToImage targetType } ]

module private UnaryImageFunctionNode =
    let functionOptions = StandardFunctionOptions.values

    let title (state: PipelineNodeState) =
        let functionName =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "function")
            |> Option.map _.Value
            |> Option.filter (fun value -> functionOptions |> List.contains value)
            |> Option.defaultValue "sqrt"

        $"{functionName}(I)"

module private VectorImageNode =
    let functionOptions = "identity" :: StandardFunctionOptions.values

    let title (state: PipelineNodeState) =
        match state.Definition.Id with
        | "VectorElement" ->
            let componentText =
                state.Parameters
                |> Seq.tryFind (fun parameter -> parameter.Key = "component")
                |> Option.map _.Value
                |> Option.filter (String.IsNullOrWhiteSpace >> not)
                |> Option.defaultValue "0"
            $"V[{componentText}]"
        | "VectorMapElements" ->
            let functionName =
                state.Parameters
                |> Seq.tryFind (fun parameter -> parameter.Key = "function")
                |> Option.map _.Value
                |> Option.filter (fun value -> functionOptions |> List.contains value)
                |> Option.defaultValue "sqrt"
            $"{functionName}(V)"
        | _ -> state.Definition.DisplayName

module private SumProjectionNode =
    let functionOptions =
        [ "Identity"
          "Abs"
          "Square"
          "SqrtAbs"
          "Log1pAbs" ]

module private ScalarImageOperationNode =
    let typeOptions = SourceImageNode.typeOptions

    let operationOptions = [ "+"; "-"; "*"; "/" ]

    let isOperation functionId =
        functionId = "ImageOpScalar"
        || functionId = "ScalarOpImage"
        || functionId = "AddNormalNoise"
        || functionId = "AddSaltAndPepperNoise"
        || functionId = "AddShotNoise"
        || functionId = "AddSpeckleNoise"

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
        |> Option.defaultValue Float64

    let ports (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let portType = PortType.numericToImage selectedType

        [ { Name = typeName
            Type = portType } ],
        [ { Name = typeName
            Type = portType } ]

    let valuePort (state: PipelineNodeState) =
        let selectedType = selectedType state

        { Name = $"a: {NumericType.toString selectedType}"
          Type = Scalar(BasicType.Numeric selectedType) }

    let title (state: PipelineNodeState) =
        let operation =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "operation")
            |> Option.map _.Value
            |> Option.filter (fun value -> operationOptions |> List.contains value)
            |> Option.defaultValue "*"

        let elementwiseOperation = "." + operation
        let valueText =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "value")
            |> Option.map (fun parameter ->
                if parameter.UseInput then
                    "a"
                else
                    parameter.Value.Trim())
            |> Option.filter (String.IsNullOrWhiteSpace >> not)
            |> Option.defaultValue "a"

        match state.Definition.Id with
        | "ImageOpScalar" -> $"I {elementwiseOperation} {valueText}"
        | "ScalarOpImage" -> $"{valueText} {elementwiseOperation} I"
        | _ -> state.Definition.DisplayName

module private ThresholdNode =
    let typeOptions = SourceImageNode.typeOptions

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
        |> Option.defaultValue Float64

    let ports (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType

        [ { Name = typeName
            Type = PortType.numericToImage selectedType } ],
        [ { Name = "UInt8"
            Type = PortType.Image UInt8 } ]

module private HighValueFilterNode =
    let typedImageSameTypeFunctionIds =
        [ "Clamp"
          "ShiftScale"
          "IntensityStretch"
          "CreatePadding"
          "Crop"
          "SmoothWMedian"
          "SmoothWBilateral"
          "GradientMagnitude"
          "SobelEdge"
          "Laplacian"
          "GrayscaleErode"
          "GrayscaleDilate"
          "GrayscaleOpening"
          "GrayscaleClosing"
          "WhiteTopHat"
          "BlackTopHat"
          "MorphologicalGradient"
          "LabelContour"
          "ChangeLabel"
          "SerialPolynomialBiasCorrect" ]
        |> Set.ofList

    let typedImageInputFunctionIds =
        [ "HistogramEqualization"
          "ImageComparison"
          "FitBiasModel"
          "FitBiasModelMasked"
          "CorrectBias"
          "CorrectBiasMasked"
          "MarchingCubes"
          "DogKeypoints"
          "SiftKeypoints"
          "LogBlobKeypoints"
          "HessianKeypoints"
          "Harris3DKeypoints"
          "Forstner3DKeypoints"
          "PhaseCongruencyKeypoints"
          "SumProjection"
          "FFT" ]
        |> Set.ofList

    let typedImageFunctionIds =
        [ "Clamp"
          "ShiftScale"
          "IntensityStretch"
          "HistogramEqualization"
          "CreatePadding"
          "Crop"
          "SmoothWMedian"
          "SmoothWBilateral"
          "GradientMagnitude"
          "SobelEdge"
          "Laplacian"
          "ImageComparison"
          "GrayscaleErode"
          "GrayscaleDilate"
          "GrayscaleOpening"
          "GrayscaleClosing"
          "WhiteTopHat"
          "BlackTopHat"
          "MorphologicalGradient"
          "OtsuThresholdFromHistogram"
          "MomentsThresholdFromHistogram"
          "MarchingCubes"
          "DogKeypoints"
          "SiftKeypoints"
          "LogBlobKeypoints"
          "HessianKeypoints"
          "Harris3DKeypoints"
          "Forstner3DKeypoints"
          "PhaseCongruencyKeypoints"
          "FitBiasModel"
          "FitBiasModelMasked"
          "CorrectBias"
          "CorrectBiasMasked"
          "SumProjection"
          "FFT"
          "SerialPolynomialBiasCorrect"
          "SerialEstTrans"
          "SerialApplyTrans"
          "SerialEstBoundingBox"
          "LabelContour"
          "ChangeLabel" ]
        |> Set.ofList

    let typeOptions = SourceImageNode.typeOptions
    let comparisonOptions = [ ">"; ">="; "<"; "<="; "="; "<>"; "!=" ]
    let maskLogicOptions = [ "and"; "or"; "xor" ]
    let histogramEstimatorOptions = [ "DKWAndHoldout"; "DKW"; "Holdout" ]
    let boolOptions = [ "false"; "true" ]
    let floatingImageTypeOptions = [ "Float32"; "Float64" ]

    let selectedType (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
        |> Option.bind (fun parameter -> NumericType.tryParse parameter.Value)
        |> Option.defaultValue Float64

    let typedImagePorts (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let portType = PortType.numericToImage selectedType

        [ { Name = typeName
            Type = portType } ],
        [ { Name = typeName
            Type = portType } ]

    let typedImageInputsWithCatalogOutputs (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let typedInput =
            { Name = typeName
              Type = PortType.numericToImage selectedType }

        let inputs =
            state.Definition.Inputs
            |> List.mapi (fun index port ->
                match index, port.Type with
                | 0, PortType.Image Number -> typedInput
                | _ -> port)

        inputs, state.Definition.Outputs

    let typedImageComparisonPorts (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let typedInput =
            { Name = typeName
              Type = PortType.numericToImage selectedType }

        [ typedInput; typedInput ], state.Definition.Outputs

    let titleFrom key fallback (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = key)
        |> Option.map _.Value
        |> Option.filter (String.IsNullOrWhiteSpace >> not)
        |> Option.defaultValue fallback

module private QuantilesNode =
    let private enabled key (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = key)
        |> Option.map (_.Value >> fun value -> value.Trim().Equals("true", StringComparison.OrdinalIgnoreCase))
        |> Option.defaultValue false

    let outputPorts (state: PipelineNodeState) =
        state.Definition.Outputs
        |> List.indexed
        |> List.filter (fun (index, _) ->
            index = 0
            || (index = 1 && enabled "useQ2" state)
            || (index = 2 && enabled "useQ3" state)
            || (index = 3 && enabled "useQ4" state)
            || (index = 4 && enabled "useQ5" state))
        |> List.map snd

module private SerialTransformNode =
    let selectedType = HighValueFilterNode.selectedType
    let typeOptions = HighValueFilterNode.floatingImageTypeOptions
    let methodOptions = [ "dogAffine"; "siftAffine"; "SSDAffine" ]

    let method (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "method")
        |> Option.map _.Value
        |> Option.defaultValue "dogAffine"

    let estimatorPorts (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType
        let imagePort =
            { Name = typeName
              Type = PortType.numericToImage selectedType }

        [ imagePort ],
        [ imagePort
          { Name = "SerialSliceManifest"
            Type = PortType.Custom "SerialSliceManifest" } ]

    let applyPorts (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType

        [ { Name = typeName
            Type = PortType.numericToImage selectedType }
          { Name = "SerialSliceManifest"
            Type = PortType.Custom "SerialSliceManifest" } ],
        [ { Name = typeName
            Type = PortType.numericToImage selectedType } ]

    let boundingBoxPorts (state: PipelineNodeState) =
        let selectedType = selectedType state
        let typeName = NumericType.toString selectedType

        [ { Name = typeName
            Type = PortType.numericToImage selectedType }
          { Name = "SerialSliceManifest"
            Type = PortType.Custom "SerialSliceManifest" } ],
        [ { Name = "SerialVolumeGeometry"
            Type = PortType.Custom "SerialVolumeGeometry" } ]

    let updateMethodParameterStates (state: PipelineNodeState) =
        if state.Definition.Id = "SerialEstTrans" then
            let isSsd = (method state).Trim().Equals("SSDAffine", StringComparison.OrdinalIgnoreCase)

            state.Parameters
            |> Seq.iter (fun parameter ->
                match parameter.Key with
                | "scale" -> parameter.IsValueEnabled <- not isSsd
                | "pixelFraction" -> parameter.IsValueEnabled <- isSsd
                | _ -> ())

module private ChartNode =
    let kindOptions =
        [ "Scatter"; "Line"; "Bar"; "Column"; "Area"; "Pie"; "Doughnut" ]

    let title (state: PipelineNodeState) =
        let kind =
            state.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "kind")
            |> Option.map _.Value
            |> Option.filter (fun value -> kindOptions |> List.contains value)
            |> Option.defaultValue "Column"

        $"Chart: {kind}"

module private PipelineNodeGeometry =
    let defaultWidth = 110.
    let defaultHeight = 48.
    let pinPadding = 20.
    let pinSpacing = 22.

module private ShowImageNode =
    let colorMapOptions =
        [ "Viridis"
          "Greys"
          "Cividis"
          "Greens"
          "Hot"
          "Jet"
          "Rainbow"
          "RdBu"
          "YlGnBu"
          "YlOrRd"
          "Blackbody"
          "Bluered"
          "Earth"
          "Electric"
          "Picnic"
          "Portland" ]

module private FiniteDiffNode =
    let directionOptions = [ "x", "0"; "y", "1"; "z", "2" ]
    let orderOptions = [ "1"; "2"; "3"; "4"; "5"; "6" ]

module private PrintNode =
    let maxInputs = 8

    let inputKey index =
        $"input{index}"

    let isInputKey (key: string) =
        key.StartsWith("input", StringComparison.Ordinal)

    let inputIndex (key: string) =
        if isInputKey key then
            let suffix = key.Substring("input".Length)
            let mutable parsed = 0

            if Int32.TryParse(suffix, &parsed) && parsed >= 1 && parsed <= maxInputs then
                Some parsed
            else
                None
        else
            None

    let inputIsVisible (state: PipelineNodeState) key =
        state.Parameters
        |> Seq.exists (fun parameter -> parameter.Key = key && parameter.UseInput)

    let activeInputIndexes (state: PipelineNodeState) =
        state.Parameters
        |> Seq.choose (fun parameter ->
            inputIndex parameter.Key
            |> Option.filter (fun _ -> parameter.UseInput))
        |> Seq.sort
        |> Seq.toList

    let placeholderForInput (state: PipelineNodeState) index =
        let inputKey = inputKey index

        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = inputKey)
        |> Option.map _.Value
        |> Option.filter (String.IsNullOrWhiteSpace >> not)
        |> Option.defaultValue inputKey
        |> fun name -> $"{{{name}}}"

    let syncFormatPlaceholders (state: PipelineNodeState) (format: string) =
        let activeInputIndexes = activeInputIndexes state
        let activeInputIndexSet = activeInputIndexes |> Set.ofList

        let inactivePlaceholders =
            [ 1..maxInputs ]
            |> List.filter (fun index -> not (activeInputIndexSet.Contains index))
            |> List.map (placeholderForInput state)
            |> Set.ofList

        let activePlaceholders =
            activeInputIndexes |> List.map (placeholderForInput state)

        let withoutInactive =
            Regex.Replace(
                format,
                @"\s*\{[^{}]+\}",
                MatchEvaluator(fun m ->
                    if inactivePlaceholders |> Set.contains (m.Value.Trim()) then
                        ""
                    else
                        m.Value))
                .Trim()

        let missing =
            activePlaceholders
            |> List.filter (fun placeholder -> not (withoutInactive.Contains(placeholder, StringComparison.Ordinal)))

        match missing with
        | [] ->
            withoutInactive
        | _ when String.IsNullOrWhiteSpace withoutInactive ->
            String.concat " " missing
        | _ ->
            withoutInactive + " " + String.concat " " missing

[<AllowNullLiteral>]
type PipelineNodeViewModel(
    state: PipelineNodeState,
    selectNode: PipelineNodeViewModel -> unit,
    moveSelectedNodesBy: PipelineNodeViewModel -> float -> float -> float -> float -> unit,
    getDrawingSize: unit -> float * float,
    markGraphDirty: unit -> unit,
    removePinConnections: IPin seq -> unit,
    refreshNodePins: PipelineNodeViewModel -> unit,
    handleTypeChange: PipelineNodeViewModel -> string -> bool) as this =
    inherit NodeViewModel()

    let mutable lastX = 0.
    let mutable lastY = 0.
    let mutable suppressGroupMove = false
    let pinSize = PipelinePinGeometry.size

    let setPinCenter x y (pin: IPin) =
        pin.X <- x
        pin.Y <- y

    let addPipelinePin x y alignment kind parameterKey (port: Port) =
        let pin = PipelinePinViewModel(alignment, port, kind, ?parameterKey = parameterKey)
        let pinName =
            match kind, port.Type with
            | ParameterInput, _ -> port.Name
            | ScalarOutput, Scalar basicType -> BasicType.toString basicType
            | ScalarOutput, _ when state.Definition.Id = "Tap" -> PortMapping.streamValueName port.Type
            | ScalarOutput, _ -> port.Name
            | ReducerOutput, _ -> port.Name
            | DataOutput, Image _ -> PortMapping.streamValueName port.Type
            | DataInput, _
            | DataOutput, _ -> port.Name

        pin.Name <- pinName

        pin.Parent <- this
        pin.Width <- pinSize
        pin.Height <- pinSize
        setPinCenter x y pin
        pin.Alignment <- alignment
        this.Pins.Add(pin :> IPin)
        pin :> IPin

    let outputKindFor functionId =
        match functionId with
        | "Scalar"
        | "FileDirectory"
        | "ScalarOp"
        | "ScalarFunction"
        | "OtsuThresholdFromHistogram"
        | "MomentsThresholdFromHistogram" -> ScalarOutput
        | "ComputeStats"
        | "SurfaceArea"
        | "Volume"
        | "PointPairDistances"
        | "AffineRegistration"
        | "FitBiasModel"
        | "FitBiasModelMasked"
        | "Expand"
        | "Write"
        | "GetChunkInfo"
        | "ComponentTranslationTable"
        | "SerialEstBoundingBox"
        | "ImHistogramData"
        | "Histogram"
        | "ObjectSizeStats"
        | "EstimateHistogram"
        | "Quantiles" -> ReducerOutput
        | _ -> DataOutput

    let outputKindForPort functionId portIndex (port: Port) =
        match functionId, portIndex, port.Type with
        | ("Read" | "ReadRandom" | "ReadRange" | "ReadSlab"), _, Custom _
        | "Write", _, Custom "StackInfo" -> ReducerOutput
        | _ -> outputKindFor functionId

    let effectivePorts () =
        match state.Definition.Id with
        | "Scalar" -> state.Definition.Inputs, [ ScalarNode.outputPort state ]
        | "FileDirectory" -> state.Definition.Inputs, state.Definition.Outputs
        | "ScalarOp" -> state.Definition.Inputs, [ ScalarOpNode.outputPort state ]
        | "ScalarFunction" -> state.Definition.Inputs, [ ScalarFunctionNode.outputPort ]
        | "Read"
        | "ReadRandom" ->
            let infoType =
                match SourceImageNode.selectedFormat state with
                | "OME-Zarr"
                | "NeXus/HDF5" -> BuiltInCatalog.chunkInfo
                | _ -> BuiltInCatalog.stackInfo
            let infoName = if infoType = BuiltInCatalog.chunkInfo then "ChunkInfo" else "StackInfo"

            state.Definition.Inputs, SourceImageNode.outputPort state :: [ { Name = infoName; Type = infoType } ]
        | "ReadSlab" ->
            state.Definition.Inputs, SourceImageNode.outputPort state :: (state.Definition.Outputs |> List.skip 1)
        | "ReadRange" ->
            state.Definition.Inputs, SourceImageNode.outputPort state :: (state.Definition.Outputs |> List.skip 1)
        | "Zero"
        | "NormalNoise"
        | "SaltAndPepperNoise"
        | "ShotNoise"
        | "SpeckleNoise"
        | "CreateByEuler2DTransform" -> state.Definition.Inputs, [ SourceImageNode.outputPort state ]
        | "Write" ->
            let infoType =
                match SourceImageNode.selectedFormat state with
                | "OME-Zarr"
                | "NeXus/HDF5" -> BuiltInCatalog.chunkInfo
                | _ -> BuiltInCatalog.stackInfo
            let infoName = if infoType = BuiltInCatalog.chunkInfo then "ChunkInfo" else "StackInfo"

            [ SourceImageNode.writeInputPort state ], [ { Name = infoName; Type = infoType } ]
        | "WriteChunks" ->
            [ SourceImageNode.writeInputPort state ], state.Definition.Outputs
        | "ImageOpImage" -> PairOperationNode.ports state
        | "Cast" -> CastNode.ports state
        | "Resize"
        | "Resample" -> HighValueFilterNode.typedImagePorts state
        | functionId when HighValueFilterNode.typedImageSameTypeFunctionIds.Contains functionId -> HighValueFilterNode.typedImagePorts state
        | "ImageComparison" -> HighValueFilterNode.typedImageComparisonPorts state
        | functionId when HighValueFilterNode.typedImageInputFunctionIds.Contains functionId -> HighValueFilterNode.typedImageInputsWithCatalogOutputs state
        | functionId when ScalarImageOperationNode.isOperation functionId -> ScalarImageOperationNode.ports state
        | "Threshold" -> ThresholdNode.ports state
        | "SerialEstTrans" -> SerialTransformNode.estimatorPorts state
        | "SerialApplyTrans" -> SerialTransformNode.applyPorts state
        | "SerialEstBoundingBox" -> SerialTransformNode.boundingBoxPorts state
        | "Quantiles" -> state.Definition.Inputs, QuantilesNode.outputPorts state
        | "Expand" ->
            state.Definition.Inputs,
            state.RecordType
            |> Option.map BuiltInCatalog.expandOutputsFor
            |> Option.defaultValue []
        | _ -> state.Definition.Inputs, state.Definition.Outputs

    let parameterPinIsVisible (parameter: PipelineParameterViewModel) =
        parameter.IsVisible
        && parameter.UseInput
        && (state.Definition.Id <> "Print" || PrintNode.inputIsVisible state parameter.Key)
        || (state.Definition.Id = "CollapseComponentLabels" && parameter.Key = "translationTable")

    let computeNodeWidth () =
        let parameterPinCount =
            state.Parameters
            |> Seq.filter parameterPinIsVisible
            |> Seq.length

        let inputs, outputs = effectivePorts ()

        let topInputCount =
            if state.Definition.Id = "Expand" then
                inputs.Length
            else
                0

        let bottomOutputCount =
            outputs
            |> List.indexed
            |> List.sumBy (fun (index, output) ->
                match outputKindForPort state.Definition.Id index output with
                | ScalarOutput
                | ReducerOutput -> 1
                | _ -> 0)
            |> fun count -> if state.Definition.Id = "Tap" then max count 1 else count

        let horizontalPinCount = max topInputCount (max parameterPinCount bottomOutputCount)
        let pinWidth = PipelineNodeGeometry.pinPadding + PipelineNodeGeometry.pinSpacing * float (max 1 horizontalPinCount)
        max PipelineNodeGeometry.defaultWidth pinWidth

    let nodeWidth () = this.Width

    let nodeHeight =
        let inputs, outputs = effectivePorts ()

        let sideInputCount =
            if state.Definition.Id = "Expand" then
                0
            else
                inputs.Length

        let sideOutputCount =
            outputs
            |> List.indexed
            |> List.sumBy (fun (index, output) ->
                match outputKindForPort state.Definition.Id index output with
                | DataOutput -> 1
                | _ -> 0)

        let portCount = max sideInputCount sideOutputCount

        let pinHeight = PipelineNodeGeometry.pinPadding + PipelineNodeGeometry.pinSpacing * float (max 1 portCount)
        max PipelineNodeGeometry.defaultHeight pinHeight

    let pinPosition length index count =
        let n = float (index + 1)
        n * length / float (count + 1)

    do
        this.Name <- state.Title
        this.Width <- computeNodeWidth ()
        this.Height <- nodeHeight
        this.Content <- PipelineNodeContent(state.Title, state, this.Width, this.Height, fun () -> selectNode this)
        this.Pins <- ObservableCollection<IPin>()

        state.Parameters
        |> Seq.iter (fun parameter ->
            parameter.PropertyChanged.Add(fun args ->
                if args.PropertyName = nameof parameter.UseInput then
                    this.SyncParameterPinVisibility()
                    this.RebuildPins()
                    if state.Definition.Id = "Print" && PrintNode.isInputKey parameter.Key then
                        state.Parameters
                        |> Seq.tryFind (fun parameter -> parameter.Key = "format")
                        |> Option.iter (fun formatParameter ->
                            formatParameter.Value <- PrintNode.syncFormatPlaceholders state formatParameter.Value)

                    if (SourceImageNode.hasInputTitle state.Definition.Id && parameter.Key = "input")
                       || (SourceImageNode.hasOutputTitle state.Definition.Id && parameter.Key = "output") then
                        state.Title <- SourceImageNode.title state
                        this.Name <- state.Title
                    elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "value" then
                        state.Title <- ScalarImageOperationNode.title state
                        this.Name <- state.Title

                    refreshNodePins this
                    markGraphDirty()
                elif state.Definition.Id = "Scalar" && parameter.Key = "value" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "FileDirectory" && parameter.Key = "value" && args.PropertyName = nameof parameter.Value then
                    state.Title <- FileDirectoryNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif SourceImageNode.hasInputTitle state.Definition.Id && parameter.Key = "input" && args.PropertyName = nameof parameter.Value then
                    state.Title <- SourceImageNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif SourceImageNode.hasOutputTitle state.Definition.Id && parameter.Key = "output" && args.PropertyName = nameof parameter.Value then
                    state.Title <- SourceImageNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "ScalarOp" && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarOpNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "ScalarFunction" && parameter.Key = "function" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarFunctionNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "ImageOpImage" && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    state.Title <- PairOperationNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "UnaryImageFunction" && parameter.Key = "function" && args.PropertyName = nameof parameter.Value then
                    state.Title <- UnaryImageFunctionNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif (state.Definition.Id = "VectorElement" && parameter.Key = "component"
                      || state.Definition.Id = "VectorMapElements" && parameter.Key = "function")
                     && args.PropertyName = nameof parameter.Value then
                    state.Title <- VectorImageNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "ImageComparison" && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    let operation = HighValueFilterNode.titleFrom "operation" ">" state
                    state.Title <- $"I {operation} J"
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "MaskLogic" && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    let operation = HighValueFilterNode.titleFrom "operation" "and" state
                    state.Title <- $"M {operation} N"
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "Chart" && parameter.Key = "kind" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ChartNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif state.Definition.Id = "Quantiles" && parameter.Key.StartsWith("useQ", StringComparison.Ordinal) && args.PropertyName = nameof parameter.Value then
                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()
                elif state.Definition.Id = "SerialEstTrans" && parameter.Key = "method" && args.PropertyName = nameof parameter.Value then
                    SerialTransformNode.updateMethodParameterStates state
                    markGraphDirty()
                elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarImageOperationNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "value" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarImageOperationNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif (state.Definition.Id = "Scalar" || state.Definition.Id = "ScalarOp" || state.Definition.Id = "Read" || state.Definition.Id = "ReadRandom" || state.Definition.Id = "EstimateHistogram" || state.Definition.Id = "ReadRange" || state.Definition.Id = "ReadSlab" || state.Definition.Id = "Zero" || state.Definition.Id = "NormalNoise" || state.Definition.Id = "SaltAndPepperNoise" || state.Definition.Id = "ShotNoise" || state.Definition.Id = "SpeckleNoise" || state.Definition.Id = "CreateByEuler2DTransform" || state.Definition.Id = "Threshold" || state.Definition.Id = "ImageOpImage" || state.Definition.Id = "Resize" || state.Definition.Id = "Resample" || state.Definition.Id = "ResampleAffineTrilinearSlices" || HighValueFilterNode.typedImageFunctionIds.Contains state.Definition.Id || ScalarImageOperationNode.isOperation state.Definition.Id) && parameter.Key = "type" && args.PropertyName = nameof parameter.Value then
                    if state.Definition.Id = "Scalar" then
                        ScalarNode.ensureValueMatchesType state
                        state.Title <- ScalarNode.title state
                        this.Name <- state.Title

                    if not (handleTypeChange this parameter.Value) then
                        this.RebuildPins()
                        refreshNodePins this

                    markGraphDirty()
                elif SourceImageNode.hasFormatParameter state.Definition.Id && parameter.Key = "suffix" && args.PropertyName = nameof parameter.Value then
                    if state.Definition.Id = "Zero"
                       || state.Definition.Id = "NormalNoise"
                       || state.Definition.Id = "SaltAndPepperNoise"
                       || state.Definition.Id = "ShotNoise"
                       || state.Definition.Id = "SpeckleNoise"
                       || state.Definition.Id = "CreateByEuler2DTransform" then
                        state.Parameters
                        |> Seq.tryFind (fun parameter -> parameter.Key = "type")
                        |> Option.iter (fun typeParameter ->
                            let supportedOptions = SourceImageNode.supportedTypeOptions state
                            let supported = supportedOptions |> Set.ofList

                            for option in typeParameter.Options do
                                option.IsEnabled <- supported |> Set.contains option.Value

                            if not (supported |> Set.contains typeParameter.Value) then
                                supportedOptions
                                |> List.tryHead
                                |> Option.iter (fun value -> typeParameter.Value <- value))

                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()
                elif (state.Definition.Id = "Read" || state.Definition.Id = "ReadRandom" || state.Definition.Id = "ReadSlab" || state.Definition.Id = "Write") && parameter.Key = "format" && args.PropertyName = nameof parameter.Value then
                    SourceImageNode.updateParameterVisibility state
                    SourceImageNode.updateReadSuffixOptionStates state
                    state.Title <- SourceImageNode.title state
                    this.Name <- state.Title
                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()
                elif state.Definition.Id = "Cast" && (parameter.Key = "sourceType" || parameter.Key = "targetType") && args.PropertyName = nameof parameter.Value then
                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()))

        this.InitializePins()
        SerialTransformNode.updateMethodParameterStates state

    member private this.RemoveConnectionsForPin(pin: IPin) =
        removePinConnections [ pin ]

    member private this.TryFindParameterPin(parameterKey: string) =
        this.Pins
        |> Seq.tryPick (function
            | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && pin.ParameterKey = parameterKey ->
                Some(pin :> IPin)
            | _ -> None)

    member private _.ParameterPinIsVisible(parameter: PipelineParameterViewModel) =
        parameterPinIsVisible parameter

    member private this.SetParameterPinVisibility(parameter: PipelineParameterViewModel, pin: IPin) =
        let isVisible = this.ParameterPinIsVisible parameter

        parameter.IsValueEnabled <- state.Definition.Id <> "Print" || not (PrintNode.isInputKey parameter.Key) || isVisible

        if isVisible then
            pin.Width <- pinSize
            pin.Height <- pinSize

            match pin with
            | :? PipelinePinViewModel as parameterPin -> parameterPin.SetActive(true)
            | _ -> ()
        else
            this.RemoveConnectionsForPin(pin)
            pin.X <- -10000.
            pin.Y <- -10000.
            // Keep hidden pins at full size; NodeEditor's centering margin can become stale after a 0x0 pin is reactivated.
            pin.Width <- pinSize
            pin.Height <- pinSize

            match pin with
            | :? PipelinePinViewModel as parameterPin -> parameterPin.SetActive(false)
            | _ -> ()

    member private this.AddParameterPin(index: int, count: int, parameter: PipelineParameterViewModel) =
        let x = pinPosition (nodeWidth ()) index count
        let port =
            if state.Definition.Id = "ScalarOp" && (parameter.Key = "a" || parameter.Key = "b") then
                ScalarOpNode.scalarPort parameter.Key state
            elif state.Definition.Id = "ScalarFunction" && parameter.Key = "a" then
                ScalarFunctionNode.scalarPort
            elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "value" then
                ScalarImageOperationNode.valuePort state
            elif state.Definition.Id = "CollapseComponentLabels" && parameter.Key = "translationTable" then
                PortMapping.customParameterPort parameter.Key "TranslationTable"
            elif state.Definition.Id = "SerialApplyTrans" && parameter.Key = "geometry" then
                PortMapping.customParameterPort parameter.Key "SerialVolumeGeometry"
            elif state.Definition.Id = "Print" && PrintNode.isInputKey parameter.Key then
                PortMapping.anyParameterPort parameter.Key
            elif state.Definition.Id = "Tap" && parameter.Key = "label" then
                PortMapping.anyParameterPort parameter.Key
            else
                PortMapping.parameterPort parameter

        let pin = addPipelinePin x 0. PinAlignment.Top ParameterInput (Some parameter.Key) port
        this.SetParameterPinVisibility(parameter, pin)

        this.TryFindParameterPin(parameter.Key)
        |> Option.iter (fun pin -> this.SetParameterPinVisibility(parameter, pin))

    member private this.SyncParameterPinVisibility() =
        let parameters = state.Parameters |> Seq.toList
        let visibleParameterIndexes =
            parameters
            |> List.indexed
            |> List.filter (fun (_, parameter) -> this.ParameterPinIsVisible parameter)
            |> List.map fst
            |> Set.ofList

        parameters
        |> List.iteri (fun index parameter ->
            match this.TryFindParameterPin(parameter.Key) with
            | Some pin ->
                this.SetParameterPinVisibility(parameter, pin)

                match visibleParameterIndexes |> Set.toList |> List.tryFindIndex ((=) index) with
                | Some visibleIndex ->
                    setPinCenter (pinPosition (nodeWidth ()) visibleIndex visibleParameterIndexes.Count) 0. pin
                | None ->
                    ()
            | None ->
                this.AddParameterPin(0, 1, parameter))

    member private this.InitializePins() =
        this.Pins.Clear()

        let inputs, outputs = effectivePorts ()

        inputs
        |> List.iteri (fun portIndex port ->
            if state.Definition.Id = "Expand" then
                addPipelinePin (pinPosition (nodeWidth ()) portIndex inputs.Length) 0. PinAlignment.Top DataInput None port |> ignore
            else
                addPipelinePin 0. (pinPosition nodeHeight portIndex inputs.Length) PinAlignment.Left DataInput None port |> ignore)

        outputs
        |> List.iteri (fun portIndex port ->
            let kind = outputKindForPort state.Definition.Id portIndex port
            let bottomOutputIndex =
                outputs
                |> List.take portIndex
                |> List.indexed
                |> List.sumBy (fun (previousIndex, output) ->
                    match outputKindForPort state.Definition.Id previousIndex output with
                    | ScalarOutput
                    | ReducerOutput -> 1
                    | _ -> 0)

            let bottomOutputCount =
                outputs
                |> List.indexed
                |> List.sumBy (fun (index, output) ->
                    match outputKindForPort state.Definition.Id index output with
                    | ScalarOutput
                    | ReducerOutput -> 1
                    | _ -> 0)

            let sideOutputIndex =
                outputs
                |> List.take portIndex
                |> List.indexed
                |> List.sumBy (fun (previousIndex, output) ->
                    match outputKindForPort state.Definition.Id previousIndex output with
                    | DataOutput -> 1
                    | _ -> 0)

            let sideOutputCount =
                outputs
                |> List.indexed
                |> List.sumBy (fun (index, output) ->
                    match outputKindForPort state.Definition.Id index output with
                    | DataOutput -> 1
                    | _ -> 0)

            let alignment =
                if kind = ScalarOutput || kind = ReducerOutput then PinAlignment.Bottom else PinAlignment.Right

            let x =
                if kind = ReducerOutput then
                    pinPosition (nodeWidth ()) bottomOutputIndex bottomOutputCount
                elif kind = ScalarOutput then
                    nodeWidth () / 2.
                else
                    nodeWidth ()

            let y =
                if kind = ScalarOutput || kind = ReducerOutput then
                    nodeHeight
                else
                    pinPosition nodeHeight sideOutputIndex sideOutputCount

            addPipelinePin x y alignment kind None port |> ignore)

        if state.Definition.Id = "Tap" then
            addPipelinePin
                (nodeWidth () / 2.)
                nodeHeight
                PinAlignment.Bottom
                ScalarOutput
                None
                { Name = "Number"
                  Type = Any }
            |> ignore

        let parameters = state.Parameters |> Seq.toList

        parameters
        |> List.iteri (fun index parameter -> this.AddParameterPin(index, parameters.Length, parameter))

        this.SyncParameterPinVisibility()

    member this.RebuildPins() =
        let updatedWidth = computeNodeWidth ()

        if this.Width <> updatedWidth then
            this.Width <- updatedWidth
            this.Content <- PipelineNodeContent(state.Title, state, this.Width, this.Height, fun () -> selectNode this)

        this.InitializePins()
        SerialTransformNode.updateMethodParameterStates state

    member _.State = state

    member this.SuppressGroupMove
        with get () = suppressGroupMove
        and set value = suppressGroupMove <- value

    member this.SyncMoveOrigin() =
        lastX <- this.X
        lastY <- this.Y

    member this.ClampToDrawing() =
        let drawingWidth, drawingHeight = getDrawingSize()
        let maxX = max 0. (drawingWidth - this.Width)
        let maxY = max 0. (drawingHeight - this.Height)

        this.X <- min maxX (max 0. this.X)
        this.Y <- min maxY (max 0. this.Y)

    override this.OnSelected() =
        base.OnSelected()
        selectNode this

    override this.OnMoved() =
        base.OnMoved()
        this.ClampToDrawing()

        let previousX = lastX
        let previousY = lastY
        let dx = this.X - previousX
        let dy = this.Y - previousY

        if not suppressGroupMove && (dx <> 0. || dy <> 0.) then
            moveSelectedNodesBy this previousX previousY dx dy

        lastX <- this.X
        lastY <- this.Y

        markGraphDirty()

type MainWindowViewModel() as this =
    inherit ViewModelBase()

    let paletteGroups = ObservableCollection<PaletteGroupViewModel>()
    let arrangeSettings = ArrangeSettingsViewModel()
    let mutable arrangeSettingsRevision = 0
    let mutable arrangeRunSerial = 0
    let mutable suppressExpandRefresh = false
    let mutable expandRefreshScheduled = false
    let mutable suppressTypePropagation = false
    let mutable suppressConnectorCollectionRefresh = false
    let mutable selectedNode: PipelineNodeViewModel = null
    let selectedNodes = HashSet<PipelineNodeViewModel>(HashIdentity.Reference)
    let mutable graphOutput = ""
    let pendingGraphOutput = StringBuilder()
    let pendingGraphOutputLock = obj()
    let mutable graphOutputFlushScheduled = false
    let mutable isRunInProgress = false
    let mutable runCancellation: CancellationTokenSource option = None
    let mutable activeRunProcess: Process option = None
    let activeRunProcessLock = obj()
    let runProjectDirectory =
        let tempRoot =
            if Directory.Exists("/private/tmp") then
                "/private/tmp"
            else
                Path.GetTempPath()

        Path.Combine(tempRoot, "StackProcessingStudioRun", string Environment.ProcessId)

    let studioWorkingDirectory = Directory.GetCurrentDirectory()
    let mutable paletteSearch = ""
    let mutable graphDirty = false
    let mutable currentGraphPath: string option = None
    let mutable resolveFileDirectoryPath: PipelineNodeViewModel -> Async<string option> =
        fun _ -> async { return None }

    let editor =
        let editor = EditorViewModel()
        editor.Factory <- MyNodeFactory()
        editor.Templates <- editor.Factory.CreateTemplates()
        editor.Drawing <- editor.Factory.CreateDrawing("StackProcessing Pipeline")
        editor

    let drawing =
        editor.Drawing :?> DrawingNodeViewModel

    do
        arrangeSettings.PropertyChanged.Add(fun _ ->
            arrangeSettingsRevision <- arrangeSettingsRevision + 1
            let revision = arrangeSettingsRevision

            async {
                do! Async.Sleep 150

                if revision = arrangeSettingsRevision then
                    this.ArrangeGraph()
            }
            |> Async.StartImmediate)

    let updatePaletteGroups () =
        paletteGroups.Clear()

        let matchingFunctions =
            BuiltInCatalog.orderedFunctions
            |> List.filter (FunctionDefinition.matches paletteSearch)

        let expandedByDefault =
            not (String.IsNullOrWhiteSpace paletteSearch)

        matchingFunctions
        |> Seq.groupBy _.Category
        |> Seq.iter (fun (category, functions) ->
            paletteGroups.Add(PaletteGroupViewModel(category, functions, expandedByDefault)))

    let createState functionId =
        let definition = BuiltInCatalog.find functionId

        let parameters =
            definition.Parameters
            |> List.map (fun parameter ->
                match definition.Id, parameter.Key with
                | "Scalar", "type" ->
                    let options =
                        ScalarNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "FileDirectory", "kind" ->
                    let options =
                        FileDirectoryNode.kindOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "FileDirectory", "value" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, canUseInput = false)
                | "ScalarOp", "operation" ->
                    let options =
                        ScalarOpNode.operationOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ScalarOp", "type" ->
                    let options =
                        ScalarOpNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ScalarFunction", "function" ->
                    let options =
                        ScalarFunctionNode.functionOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("Read" | "ReadRandom" | "ReadSlab"), "format" ->
                    let options =
                        (if functionId = "ReadSlab" then SourceImageNode.readSlabFormatOptions else SourceImageNode.readFormatOptions)
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Write", "format" ->
                    let options =
                        SourceImageNode.writeFormatOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("Read" | "ReadRandom" | "ReadSlab"), "type" ->
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("EstimateHistogram" | "ReadRange"), "type" ->
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | functionId, "type" when SourceImageNode.isSyntheticSource functionId ->
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ResampleAffineTrilinearSlices", "type" ->
                    let defaultSuffix =
                        definition.Parameters
                        |> List.tryFind (fun parameter -> parameter.Key = "suffix")
                        |> Option.map _.DefaultValue
                        |> Option.defaultValue ".tiff"
                    let supported =
                        ImageFileFormat.supportedTypes defaultSuffix
                        |> List.map NumericType.toString
                        |> Set.ofList
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, supported |> Set.contains value))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("SmoothWGauss" | "Convolve"), "outputRegionMode" ->
                    let options =
                        [ "None"; "Valid"; "Same" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("SmoothWGauss" | "Convolve"), "boundaryCondition" ->
                    let options =
                        [ "None"; "ZeroPad"; "PerodicPad"; "ZeroFluxNeumannPad" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("Resize" | "Resample"), "interpolation" ->
                    let options =
                        [ "Linear"; "NearestNeighbor" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("Resize" | "Resample"), "type" ->
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | functionId, "suffix" when SourceImageNode.hasFormatParameter functionId ->
                    let options =
                        SourceImageNode.suffixOptionsFor functionId
                        |> List.map (fun (label, value) -> ParameterOptionViewModel(label, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("WritePointSet" | "WriteMatrix"), "suffix" ->
                    let options =
                        [ "CSV", ".csv" ]
                        |> List.map (fun (label, value) -> ParameterOptionViewModel(label, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "WriteCSV", "dataKind" ->
                    let options =
                        [ "PointSet"; "Matrix"; "Histogram" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Cast", ("sourceType" | "targetType") ->
                    let options =
                        CastNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ImageOpImage", "type" ->
                    let options =
                        PairOperationNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "FFT", "type" ->
                    let options =
                        PairOperationNode.typeOptions
                        |> List.filter ((<>) (NumericType.toString Complex))
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ImageOpImage", "operation" ->
                    let options =
                        PairOperationNode.operationOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "UnaryImageFunction", "function" ->
                    let options =
                        UnaryImageFunctionNode.functionOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "VectorMapElements", "function" ->
                    let options =
                        VectorImageNode.functionOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "PCA", "components" ->
                    let options =
                        [ 2 .. 8 ]
                        |> List.map (fun value ->
                            let value = string value
                            ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "SumProjection", "function" ->
                    let options =
                        SumProjectionNode.functionOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ImageComparison", "operation" ->
                    let options =
                        HighValueFilterNode.comparisonOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "MaskLogic", "operation" ->
                    let options =
                        HighValueFilterNode.maskLogicOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "EstimateHistogram", "estimator" ->
                    let options =
                        HighValueFilterNode.histogramEstimatorOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("BinaryContour" | "LabelContour"), "fullyConnected" ->
                    let options =
                        HighValueFilterNode.boolOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("StreamConnectedObjects" | "RemoveSmallObjects" | "FillSmallHoles"), "connectivity" ->
                    let options =
                        [ "Six"; "TwentySix" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "HessianKeypoints", "responseKind" ->
                    let options =
                        [ "Blob"; "Tube"; "Sheet" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Quantiles", key when key.StartsWith("useQ", StringComparison.Ordinal) ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, canUseInput = false)
                | ("CorrectBias" | "CorrectBiasMasked"), "model" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, forceUseInput = true)
                | ("Quantiles" | "HistogramEqualization"), "histogram" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, forceUseInput = true)
                | ("SerialPolynomialBiasCorrect" | "SerialEstTrans" | "SerialApplyTrans" | "SerialEstBoundingBox"), "type" ->
                    let options =
                        SerialTransformNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "SerialEstTrans", "method" ->
                    let options =
                        SerialTransformNode.methodOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | functionId, "type" when HighValueFilterNode.typedImageFunctionIds.Contains functionId ->
                    let options =
                        HighValueFilterNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | functionId, "operation" when ScalarImageOperationNode.isOperation functionId ->
                    let options =
                        ScalarImageOperationNode.operationOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | functionId, "type" when ScalarImageOperationNode.isOperation functionId ->
                    let options =
                        ScalarImageOperationNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Threshold", "type" ->
                    let options =
                        ThresholdNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "FiniteDiff", "direction" ->
                    let options =
                        FiniteDiffNode.directionOptions
                        |> List.map (fun (label, value) -> ParameterOptionViewModel(label, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "FiniteDiff", "order" ->
                    let options =
                        FiniteDiffNode.orderOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Chart", "kind" ->
                    let options =
                        ChartNode.kindOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ShowImage", "colorMap" ->
                    let options =
                        ShowImageNode.colorMapOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Print", "format" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, canUseInput = false)
                | "Print", key when PrintNode.isInputKey key ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, initialUseInput = (key = "input1"))
                | "Chart", "input" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, forceUseInput = true)
                | _ ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type))

        let state = PipelineNodeState(definition, parameters)

        if definition.Id = "Scalar" then
            state.Title <- ScalarNode.title state
        elif definition.Id = "FileDirectory" then
            state.Title <- FileDirectoryNode.title state
        elif definition.Id = "ScalarOp" then
            state.Title <- ScalarOpNode.title state
        elif definition.Id = "ScalarFunction" then
            state.Title <- ScalarFunctionNode.title state
        elif SourceImageNode.hasInputTitle definition.Id then
            state.Title <- SourceImageNode.title state
        elif SourceImageNode.hasOutputTitle definition.Id then
            state.Title <- SourceImageNode.title state
        elif definition.Id = "ImageOpImage" then
            state.Title <- PairOperationNode.title state
        elif definition.Id = "UnaryImageFunction" then
            state.Title <- UnaryImageFunctionNode.title state
        elif definition.Id = "VectorElement" || definition.Id = "VectorMapElements" then
            state.Title <- VectorImageNode.title state
        elif definition.Id = "ImageComparison" then
            let operation = HighValueFilterNode.titleFrom "operation" ">" state
            state.Title <- $"I {operation} J"
        elif definition.Id = "MaskLogic" then
            let operation = HighValueFilterNode.titleFrom "operation" "and" state
            state.Title <- $"M {operation} N"
        elif definition.Id = "Chart" then
            state.Title <- ChartNode.title state
        elif definition.Id = "ImageOpScalar" || definition.Id = "ScalarOpImage" then
            state.Title <- ScalarImageOperationNode.title state

        SerialTransformNode.updateMethodParameterStates state
        SourceImageNode.updateParameterVisibility state
        state

    let watchState (state: PipelineNodeState) =
        state.Parameters
        |> Seq.iter (fun parameter ->
            parameter.PropertyChanged.Add(fun _ -> this.MarkGraphDirty()))

    let pipelineNodes () =
        drawing.Nodes
        |> Seq.choose (function
            | :? PipelineNodeViewModel as node -> Some node
            | _ -> None)

    let pipelineStates () =
        pipelineNodes ()
        |> Seq.map _.State

    let fileDirectoryNodes () =
        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "FileDirectory")
        |> Seq.toArray

    let clearSelectedNodes () =
        selectedNodes
        |> Seq.toArray
        |> Array.iter (fun node -> node.State.IsSelected <- false)

        selectedNodes.Clear()

    let addSelectedNode (node: PipelineNodeViewModel) =
        if not (isNull node) && selectedNodes.Add node then
            node.State.IsSelected <- true

    let selectOnlyNode (node: PipelineNodeViewModel) =
        clearSelectedNodes()

        if not (isNull node) then
            addSelectedNode node

    let clampSelectionDelta (selected: PipelineNodeViewModel array) dx dy =
        if selected.Length = 0 then
            0., 0.
        else
            let left = selected |> Array.map _.X |> Array.min
            let top = selected |> Array.map _.Y |> Array.min
            let right = selected |> Array.map (fun node -> node.X + node.Width) |> Array.max
            let bottom = selected |> Array.map (fun node -> node.Y + node.Height) |> Array.max

            let minDx = -left
            let maxDx = drawing.Width - right
            let minDy = -top
            let maxDy = drawing.Height - bottom

            let clampDelta minDelta maxDelta delta =
                if maxDelta < minDelta then
                    0.
                else
                    min maxDelta (max minDelta delta)

            clampDelta minDx maxDx dx, clampDelta minDy maxDy dy

    let applySelectionDelta (selected: PipelineNodeViewModel array) dx dy =
        if selected.Length > 0 && (dx <> 0. || dy <> 0.) then
            selected
            |> Array.iter (fun node ->
                node.SuppressGroupMove <- true

                try
                    node.X <- node.X + dx
                    node.Y <- node.Y + dy
                    node.SyncMoveOrigin()
                finally
                    node.SuppressGroupMove <- false)

            this.MarkGraphDirty()

    let moveSelectedNodesBy (movedNode: PipelineNodeViewModel) previousX previousY dx dy =
        if selectedNodes.Contains movedNode && selectedNodes.Count > 1 then
            movedNode.SuppressGroupMove <- true

            try
                movedNode.X <- previousX
                movedNode.Y <- previousY

                let selected = selectedNodes |> Seq.toArray
                let clampedDx, clampedDy = clampSelectionDelta selected dx dy
                applySelectionDelta selected clampedDx clampedDy
            finally
                movedNode.SuppressGroupMove <- false

    let isNumberImagePin (pin: IPin) =
        match pin with
        | :? PipelinePinViewModel as pipelinePin ->
            match pipelinePin.Port.Type with
            | Any ->
                true
            | Image Number -> true
            | _ -> false
        | _ -> false

    let isAdaptableImageInputPin (pin: IPin) =
        match pin with
        | :? PipelinePinViewModel as pipelinePin when pipelinePin.Kind = DataInput ->
            match pipelinePin.Parent, pipelinePin.Port.Type with
            | :? PipelineNodeViewModel as node, Image _ ->
                node.State.Definition.Id = "Cast"
                || (node.State.Parameters |> Seq.exists (fun parameter -> parameter.Key = "type"))
            | _ ->
                false
        | _ ->
            false

    let hasConnectionRequiringFixedDataOutput (node: PipelineNodeViewModel) =
        node.Pins
        |> Seq.exists (function
            | :? PipelinePinViewModel as pin when pin.Kind = DataOutput ->
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    Object.ReferenceEquals(connector.Start, pin)
                    && not (isNumberImagePin connector.End)
                    && not (isAdaptableImageInputPin connector.End))
            | _ -> false)

    let refreshScalarTypeOptions () =
        let connectedScalarType (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.tryPick (function
                | :? PipelinePinViewModel as pin when pin.Kind = ScalarOutput ->
                    drawing.Connectors
                    |> Seq.tryFind (fun connector -> Object.ReferenceEquals(connector.Start, pin))
                    |> Option.bind (fun connector ->
                        match connector.End with
                        | :? PipelinePinViewModel as endPin ->
                            match endPin.Port.Type with
                            | Scalar basicType -> Some(BasicType.toString basicType)
                            | _ -> None
                        | _ -> None)
                | _ -> None)

        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "Scalar")
        |> Seq.iter (fun node ->
            let allowedType = connectedScalarType node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <-
                        match allowedType with
                        | Some allowed -> option.Value = allowed
                        | None -> true

                match allowedType with
                | Some allowed when parameter.Value <> allowed ->
                    parameter.Value <- allowed
                | _ -> ()))

    let refreshScalarOpTypeOptions () =
        let connectedScalarType (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.tryPick (function
                | :? PipelinePinViewModel as pin when pin.Kind = ScalarOutput ->
                    drawing.Connectors
                    |> Seq.tryFind (fun connector -> Object.ReferenceEquals(connector.Start, pin))
                    |> Option.bind (fun connector ->
                        match connector.End with
                        | :? PipelinePinViewModel as endPin ->
                            match endPin.Port.Type with
                            | Scalar basicType -> Some(BasicType.toString basicType)
                            | _ -> None
                        | _ -> None)
                | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && (pin.ParameterKey = "a" || pin.ParameterKey = "b") ->
                    drawing.Connectors
                    |> Seq.tryFind (fun connector -> Object.ReferenceEquals(connector.End, pin))
                    |> Option.bind (fun connector ->
                        match connector.Start with
                        | :? PipelinePinViewModel as startPin ->
                            match startPin.Port.Type with
                            | Scalar basicType -> Some(BasicType.toString basicType)
                            | _ -> None
                        | _ -> None)
                | _ -> None)

        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "ScalarOp")
        |> Seq.iter (fun node ->
            let allowedType = connectedScalarType node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <-
                        match allowedType with
                        | Some allowed -> option.Value = allowed
                        | None -> true

                match allowedType with
                | Some allowed when parameter.Value <> allowed ->
                    parameter.Value <- allowed
                | _ -> ()))

    let refreshSourceImageTypeOptions () =
        pipelineNodes ()
        |> Seq.filter (fun node ->
            node.State.Definition.Id = "Read"
            || node.State.Definition.Id = "ReadRandom"
            || node.State.Definition.Id = "ReadRange"
            || node.State.Definition.Id = "ReadSlab"
            || node.State.Definition.Id = "Zero"
            || node.State.Definition.Id = "NormalNoise"
            || node.State.Definition.Id = "SaltAndPepperNoise"
            || node.State.Definition.Id = "ShotNoise"
            || node.State.Definition.Id = "SpeckleNoise"
            || node.State.Definition.Id = "CreateByEuler2DTransform")
        |> Seq.iter (fun node ->
            let isConnected = hasConnectionRequiringFixedDataOutput node
            let supportedTypes =
                SourceImageNode.supportedTypes node.State
                |> List.map NumericType.toString
                |> Set.ofList

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    let supported = supportedTypes |> Set.contains option.Value
                    option.IsEnabled <-
                        if SourceImageNode.isReadSource node.State.Definition.Id || SourceImageNode.isSyntheticSource node.State.Definition.Id then
                            supported
                        else
                            supported && (not isConnected || option.Value = parameter.Value)

                if not (supportedTypes |> Set.contains parameter.Value)
                   && (not isConnected || SourceImageNode.isReadSource node.State.Definition.Id || SourceImageNode.isSyntheticSource node.State.Definition.Id) then
                    SourceImageNode.supportedTypeOptions node.State
                    |> List.tryHead
                    |> Option.iter (fun value -> parameter.Value <- value)))

    let connectedWriteInputType (node: PipelineNodeViewModel) =
        node.Pins
        |> Seq.tryPick (function
            | :? PipelinePinViewModel as inputPin when inputPin.Kind = DataInput ->
                drawing.Connectors
                |> Seq.tryPick (fun connector ->
                    if Object.ReferenceEquals(connector.End, inputPin) then
                        match connector.Start with
                        | :? PipelinePinViewModel as outputPin ->
                            match outputPin.Port.Type with
                            | Image Number -> None
                            | Image numericType -> Some numericType
                            | _ -> None
                        | _ ->
                            None
                    else
                        None)
            | _ ->
                None)

    let refreshWriteInputPin (node: PipelineNodeViewModel) =
        if SourceImageNode.hasOutputTitle node.State.Definition.Id then
            let staticPort = SourceImageNode.writeInputPort node.State
            let port =
                if staticPort.Type <> Image Number then
                    staticPort
                else
                    connectedWriteInputType node
                    |> Option.map (fun numericType ->
                        { Name = NumericType.toString numericType
                          Type = PortType.numericToImage numericType })
                    |> Option.defaultValue staticPort

            node.Pins
            |> Seq.tryPick (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                | _ -> None)
            |> Option.iter (fun pin ->
                pin.Port <- port
                pin.Name <- port.Name)

    let refreshConnectedImageInputPins () =
        let dynamicInputNodeIds = Set.ofList [ "ImHistogram"; "ImHistogramData" ]

        let connectedImageInputType (inputPin: PipelinePinViewModel) =
            drawing.Connectors
            |> Seq.tryPick (fun connector ->
                if Object.ReferenceEquals(connector.End, inputPin) then
                    match connector.Start with
                    | :? PipelinePinViewModel as outputPin ->
                        match outputPin.Port.Type with
                        | Image Number -> None
                        | Image numericType -> Some numericType
                        | _ -> None
                    | _ ->
                        None
                else
                    None)

        pipelineNodes ()
        |> Seq.filter (fun node -> dynamicInputNodeIds |> Set.contains node.State.Definition.Id)
        |> Seq.iter (fun node ->
            let defaultInputs = node.State.Definition.Inputs

            node.Pins
            |> Seq.choose (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput -> Some pin
                | _ -> None)
            |> Seq.iteri (fun index pin ->
                let defaultPort =
                    defaultInputs
                    |> List.tryItem index
                    |> Option.defaultValue pin.Port

                let port =
                    connectedImageInputType pin
                    |> Option.map (fun numericType ->
                        { Name = NumericType.toString numericType
                          Type = PortType.numericToImage numericType })
                    |> Option.defaultValue defaultPort

                pin.Port <- port
                pin.Name <- port.Name))

    let refreshImageFormatOptions () =
        pipelineNodes ()
        |> Seq.filter (fun node -> SourceImageNode.hasFormatParameter node.State.Definition.Id || SourceImageNode.hasOutputTitle node.State.Definition.Id)
        |> Seq.iter (fun node ->
            refreshWriteInputPin node

            let suffixOptionIsEnabled suffix =
                if SourceImageNode.hasInputTitle node.State.Definition.Id || node.State.Definition.Id = "Write" then
                    match node.State.Definition.Id, SourceImageNode.selectedFormat node.State with
                    | "Read", "Volume file" -> suffix = ".tiff"
                    | "Write", "Volume file" -> suffix = ".tiff"
                    | _ -> true
                else
                    let requiredType =
                        if SourceImageNode.hasOutputTitle node.State.Definition.Id then
                            connectedWriteInputType node
                        else
                            None

                    requiredType
                    |> Option.forall (ImageFileFormat.supports suffix)

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "suffix")
            |> Option.iter (fun parameter ->
                if SourceImageNode.hasInputTitle node.State.Definition.Id || node.State.Definition.Id = "Write" then
                    SourceImageNode.updateReadSuffixOptionStates node.State
                else
                    for option in parameter.Options do
                        option.IsEnabled <- suffixOptionIsEnabled option.Value

                    if parameter.Options |> Seq.exists (fun option -> option.Value = parameter.Value && option.IsEnabled) |> not then
                        parameter.Options
                        |> Seq.tryFind _.IsEnabled
                        |> Option.iter (fun option -> parameter.Value <- option.Value)))

    let refreshCastTypeOptions () =
        let hasInputConnection (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.exists (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                | _ -> false)

        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "Cast")
        |> Seq.iter (fun node ->
            let sourceConnected = hasInputConnection node
            let targetConnected = hasConnectionRequiringFixedDataOutput node

            node.State.Parameters
            |> Seq.iter (fun parameter ->
                let isConnectedType =
                    match parameter.Key with
                    | "sourceType" -> Some sourceConnected
                    | "targetType" -> Some targetConnected
                    | _ -> None

                match isConnectedType with
                | Some isConnected ->
                    for option in parameter.Options do
                        option.IsEnabled <- not isConnected || option.Value = parameter.Value
                | None -> ()))

    let refreshPairOperationTypeOptions () =
        let hasConnection (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.exists (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                | _ -> false)

        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "ImageOpImage")
        |> Seq.iter (fun node ->
            let isConnected = hasConnection node || hasConnectionRequiringFixedDataOutput node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <- not isConnected || option.Value = parameter.Value))

    let refreshScalarImageOperationTypeOptions () =
        let hasConnection (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.exists (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput || (pin.Kind = ParameterInput && pin.ParameterKey = "value") ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                | _ -> false)

        pipelineNodes ()
        |> Seq.filter (fun node -> ScalarImageOperationNode.isOperation node.State.Definition.Id)
        |> Seq.iter (fun node ->
            let isConnected = hasConnection node || hasConnectionRequiringFixedDataOutput node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <- not isConnected || option.Value = parameter.Value))

    let refreshThresholdTypeOptions () =
        let hasInputConnection (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.exists (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                | _ -> false)

        pipelineNodes ()
        |> Seq.filter (fun node -> node.State.Definition.Id = "Threshold")
        |> Seq.iter (fun node ->
            let isConnected = hasInputConnection node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <- not isConnected || option.Value = parameter.Value))

    let refreshSerialTransformTypeOptions () =
        let hasConstrainedConnection (node: PipelineNodeViewModel) =
            node.Pins
            |> Seq.exists (function
                | :? PipelinePinViewModel as pin when pin.Kind = DataInput && pin.Port.Type <> PortType.Custom "SerialSliceManifest" ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.End, pin))
                | :? PipelinePinViewModel as pin when pin.Kind = DataOutput ->
                    drawing.Connectors
                    |> Seq.exists (fun connector -> Object.ReferenceEquals(connector.Start, pin))
                | _ -> false)

        pipelineNodes ()
        |> Seq.filter (fun node ->
            node.State.Definition.Id = "SerialEstTrans"
            || node.State.Definition.Id = "SerialApplyTrans"
            || node.State.Definition.Id = "SerialEstBoundingBox")
        |> Seq.iter (fun node ->
            let isConnected = hasConstrainedConnection node

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "type")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <- not isConnected || option.Value = parameter.Value))

    let parameterValues (state: PipelineNodeState) =
        state.Parameters
        |> Seq.map (fun parameter ->
            { Key = parameter.Key
              Value = parameter.Value
              UseInput = parameter.UseInput })
        |> Seq.toArray

    let stripFSharpNumericLiteralSuffix (value: string) =
        let trimmed = value.Trim()

        let isNumericBody (body: string) =
            not (String.IsNullOrWhiteSpace body)
            && body |> Seq.forall (fun c -> Char.IsDigit c || c = '.' || c = '-' || c = '+')

        let suffixes = [ "uy"; "us"; "ul"; "y"; "s"; "u"; "l"; "f" ]

        suffixes
        |> List.tryFind (fun suffix ->
            trimmed.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)
            && isNumericBody (trimmed.Substring(0, trimmed.Length - suffix.Length)))
        |> Option.map (fun suffix -> trimmed.Substring(0, trimmed.Length - suffix.Length))
        |> Option.defaultValue (
            if String.Equals(trimmed, "System.Numerics.Complex.One", StringComparison.Ordinal) then
                "1.0"
            elif String.Equals(trimmed, "System.Math.E", StringComparison.Ordinal) then
                "e"
            elif String.Equals(trimmed, "System.Math.PI", StringComparison.Ordinal) then
                "pi"
            else
                value)

    let userFacingSavedValue (state: PipelineNodeState) (parameter: PipelineParameterViewModel) (savedValue: string) =
        match parameter.ParameterType with
        | BasicType.Numeric _ ->
            stripFSharpNumericLiteralSuffix savedValue
        | BasicType.String when state.Definition.Id = "PermuteAxes" && parameter.Key = "axes" ->
            savedValue.Replace("u", "").Replace("U", "")
        | _ ->
            savedValue

    let setParameterValues (state: PipelineNodeState) (parameters: SavedParameter array) =
        let values =
            parameters
            |> Seq.map (fun parameter -> parameter.Key, parameter)
            |> Map.ofSeq

        for parameter in state.Parameters do
            match values |> Map.tryFind parameter.Key with
            | Some savedParameter ->
                parameter.Value <- userFacingSavedValue state parameter savedParameter.Value
                parameter.UseInput <- savedParameter.UseInput
            | None -> ()

    let tryPin alignment (node: INode) =
        node.Pins
        |> Seq.tryFind (fun pin -> pin.Alignment = alignment)

    let pinByKindIndex kind index (node: PipelineNodeViewModel) =
        if kind = ParameterInput then
            node.State.Parameters
            |> Seq.tryItem index
            |> Option.bind (fun parameter ->
                node.Pins
                |> Seq.choose (function
                    | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && pin.ParameterKey = parameter.Key -> Some pin
                    | _ -> None)
                |> Seq.tryHead)
        else
            node.Pins
            |> Seq.choose (function
                | :? PipelinePinViewModel as pin when pin.Kind = kind -> Some pin
                | _ -> None)
            |> Seq.tryItem index

    let pinIndexByKind kind (pin: IPin) (node: PipelineNodeViewModel) =
        match pin with
        | :? PipelinePinViewModel as parameterPin when kind = ParameterInput ->
            node.State.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = parameterPin.ParameterKey)
        | _ ->
            node.Pins
            |> Seq.choose (function
                | :? PipelinePinViewModel as candidate when candidate.Kind = kind -> Some candidate
                | _ -> None)
            |> Seq.mapi (fun index candidate -> index, candidate)
            |> Seq.tryFind (fun (_, candidate) -> Object.ReferenceEquals(candidate, pin))
            |> Option.map fst

    let canConnectPins (startPin: IPin) (endPin: IPin) =
        let concreteImageType (portType: PortType) =
            match portType with
            | Image Number -> None
            | Image numericType -> Some numericType
            | _ -> None

        let assignedParameterInputType (inputPin: PipelinePinViewModel) =
            drawing.Connectors
            |> Seq.tryPick (fun connector ->
                if Object.ReferenceEquals(connector.End, inputPin) then
                    match connector.Start with
                    | :? PipelinePinViewModel as startPin -> Some startPin.Port.Type
                    | _ -> None
                else
                    None)

        let tapConcreteTypes (tapNode: INode) =
            drawing.Connectors
            |> Seq.choose (fun connector ->
                match connector.Start, connector.End with
                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                    when Object.ReferenceEquals(startPin.Parent, tapNode)
                         && startPin.Kind = DataOutput ->
                    concreteImageType endPin.Port.Type
                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                    when Object.ReferenceEquals(endPin.Parent, tapNode)
                         && endPin.Kind = DataInput ->
                    concreteImageType startPin.Port.Type
                | _ ->
                    None)
            |> Seq.distinct
            |> Seq.toArray

        match startPin, endPin with
        | :? PipelinePinViewModel as outputPin, (:? PipelinePinViewModel as inputPin)
            when outputPin.IsOutput && outputPin.IsActive && inputPin.IsInput && inputPin.IsActive ->
            let isPrintParameter =
                match inputPin.Parent with
                | :? PipelineNodeViewModel as node ->
                    node.State.Definition.Id = "Print" && inputPin.Kind = ParameterInput
                | _ ->
                    false

            if isPrintParameter && (outputPin.Kind = ScalarOutput || outputPin.Kind = ReducerOutput) then
                assignedParameterInputType inputPin
                |> Option.forall (PortType.canConnect outputPin.Port.Type)
            elif outputPin.Kind = ReducerOutput
                 && inputPin.Kind = DataInput
                 && (match inputPin.Parent with
                     | :? PipelineNodeViewModel as node -> node.State.Definition.Id = "Expand"
                     | _ -> false) then
                PortType.canConnect outputPin.Port.Type inputPin.Port.Type
            elif (outputPin.Kind = DataOutput || outputPin.Kind = ReducerOutput) && inputPin.Kind = DataInput then
                let baseCompatible = PortType.canConnect outputPin.Port.Type inputPin.Port.Type
                let formatCompatible =
                    match inputPin.Parent, outputPin.Port.Type with
                    | :? PipelineNodeViewModel as inputNode, Image Number
                        when SourceImageNode.hasOutputTitle inputNode.State.Definition.Id ->
                        true
                    | :? PipelineNodeViewModel as inputNode, Image numericType
                        when SourceImageNode.hasOutputTitle inputNode.State.Definition.Id ->
                        SourceImageNode.supportsOutputType inputNode.State numericType
                    | _ ->
                        true

                let isTapNode (node: INode) =
                    match node with
                    | :? PipelineNodeViewModel as pipelineNode -> pipelineNode.State.Definition.Id = "Tap"
                    | _ -> false

                let candidateTypesForTap tapNode =
                    seq {
                        yield! tapConcreteTypes tapNode

                        if Object.ReferenceEquals(outputPin.Parent, tapNode) then
                            match concreteImageType inputPin.Port.Type with
                            | Some numericType -> yield numericType
                            | None -> ()

                        if Object.ReferenceEquals(inputPin.Parent, tapNode) then
                            match concreteImageType outputPin.Port.Type with
                            | Some numericType -> yield numericType
                            | None -> ()
                    }
                    |> Seq.distinct
                    |> Seq.toArray

                let hasTapEndpoint = isTapNode outputPin.Parent || isTapNode inputPin.Parent

                if hasTapEndpoint then
                    [| outputPin.Parent; inputPin.Parent |]
                    |> Array.filter isTapNode
                    |> Array.distinctBy RuntimeHelpers.GetHashCode
                    |> Array.forall (fun tapNode -> (candidateTypesForTap tapNode).Length <= 1)
                    && formatCompatible
                else
                    baseCompatible && formatCompatible
            elif (outputPin.Kind = ScalarOutput || outputPin.Kind = ReducerOutput) && inputPin.Kind = ParameterInput then
                PortType.canConnect outputPin.Port.Type inputPin.Port.Type
            else
                false
        | _ -> false

    let connectorOrientation (startPin: IPin) (endPin: IPin) =
        match startPin, endPin with
        | (:? PipelinePinViewModel as outputPin), (:? PipelinePinViewModel as inputPin)
            when outputPin.Kind = ScalarOutput || inputPin.Kind = ParameterInput ->
            ConnectorOrientation.Vertical
        | (:? PipelinePinViewModel as outputPin), (:? PipelinePinViewModel as inputPin)
            when outputPin.Kind = ReducerOutput
                 && inputPin.Kind = DataInput
                 && (match inputPin.Parent with
                     | :? PipelineNodeViewModel as node -> node.State.Definition.Id = "Expand"
                     | _ -> false) ->
            ConnectorOrientation.Vertical
        | _ ->
            ConnectorOrientation.Horizontal

    let removePinConnections (pins: IPin seq) =
        let pins = pins |> Seq.toArray

        let connectors =
            drawing.Connectors
            |> Seq.filter (fun connector ->
                pins
                |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
            |> Seq.toArray

        for connector in connectors do
            drawing.Connectors.Remove(connector) |> ignore

    let refreshNodePins (node: PipelineNodeViewModel) =
        if drawing.Nodes.Contains(node :> INode) then
            let dynamicPinIndex (pin: PipelinePinViewModel) =
                match pin.Kind with
                | ParameterInput ->
                    node.State.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = pin.ParameterKey)
                | DataInput
                | DataOutput ->
                    match pin.Kind, pin.Port.Name with
                    | DataInput, name when name.StartsWith("J:", StringComparison.Ordinal) -> Some 1
                    | DataInput, name when name.StartsWith("K:", StringComparison.Ordinal) -> Some 2
                    | DataInput, name when name.StartsWith("L:", StringComparison.Ordinal) -> Some 3
                    | _ -> Some 0
                | ScalarOutput ->
                    Some 0
                | ReducerOutput ->
                    node.Pins
                    |> Seq.choose (function
                        | :? PipelinePinViewModel as outputPin when outputPin.Kind = ReducerOutput -> Some outputPin
                        | _ -> None)
                    |> Seq.tryFindIndex (fun outputPin -> outputPin.Port.Name = pin.Port.Name)

            let nodeEndpoint (pin: IPin) =
                match pin with
                | :? PipelinePinViewModel as pipelinePin when Object.ReferenceEquals(pipelinePin.Parent, node) ->
                    true, dynamicPinIndex pipelinePin |> Option.map (fun index -> pipelinePin.Kind, index)
                | _ -> false, None

            let connectors =
                drawing.Connectors
                |> Seq.filter (fun connector -> Object.ReferenceEquals(connector.Start.Parent, node) || Object.ReferenceEquals(connector.End.Parent, node))
                |> Seq.toArray

            let connectorSnapshots =
                connectors
                |> Array.map (fun connector ->
                    let startBelongsToNode, startEndpoint = nodeEndpoint connector.Start
                    let endBelongsToNode, endEndpoint = nodeEndpoint connector.End
                    connector, startBelongsToNode, startEndpoint, endBelongsToNode, endEndpoint)

            for connector in connectors do
                drawing.Connectors.Remove(connector) |> ignore

            for connector, startBelongsToNode, startEndpoint, endBelongsToNode, endEndpoint in connectorSnapshots do
                let startReplacement =
                    startEndpoint
                    |> Option.bind (fun (kind, index) -> pinByKindIndex kind index node)

                let endReplacement =
                    endEndpoint
                    |> Option.bind (fun (kind, index) -> pinByKindIndex kind index node)

                let startStillExists =
                    if startBelongsToNode then
                        startReplacement.IsSome
                    else
                        true

                let endStillExists =
                    if endBelongsToNode then
                        endReplacement.IsSome
                    else
                        true

                startReplacement
                |> Option.iter (fun pin -> connector.Start <- pin)

                endReplacement
                |> Option.iter (fun pin -> connector.End <- pin)

                if startStillExists && endStillExists && canConnectPins connector.Start connector.End then
                    drawing.Connectors.Add(connector) |> ignore

    let handleTypeChange (sourceNode: PipelineNodeViewModel) (typeValue: string) =
        if suppressTypePropagation then
            true
        else
            suppressTypePropagation <- true
            suppressConnectorCollectionRefresh <- true

            try
                let parameterCanUseValue (parameter: PipelineParameterViewModel) =
                    parameter.Options.Count = 0
                    || (parameter.Options |> Seq.exists (fun option -> option.Value = typeValue))

                let nodeHasTypeParameter (node: PipelineNodeViewModel) =
                    node.State.Parameters
                    |> Seq.exists (fun parameter -> parameter.Key = "type")

                let nodeCanUseType (node: PipelineNodeViewModel) =
                    node.State.Parameters
                    |> Seq.tryFind (fun parameter -> parameter.Key = "type")
                    |> Option.exists parameterCanUseValue

                let writerAcceptsType (node: PipelineNodeViewModel) =
                    match NumericType.tryParse typeValue with
                    | Some numericType when SourceImageNode.hasOutputTitle node.State.Definition.Id ->
                        SourceImageNode.supportsOutputType node.State numericType
                    | _ ->
                        true

                let portTypeForValue =
                    NumericType.tryParse typeValue
                    |> Option.map PortType.numericToImage

                let fixedInputAcceptsType (pin: PipelinePinViewModel) =
                    match pin.Parent with
                    | :? PipelineNodeViewModel as node when SourceImageNode.hasOutputTitle node.State.Definition.Id ->
                        writerAcceptsType node
                    | _ ->
                        portTypeForValue
                        |> Option.exists (fun outputType -> PortType.canConnect outputType pin.Port.Type)

                let rec branchCanAcceptType (node: PipelineNodeViewModel) (visited: HashSet<PipelineNodeViewModel>) =
                    if node.State.Definition.Id = "Cast" then
                        true
                    elif not (visited.Add node) then
                        true
                    elif SourceImageNode.hasOutputTitle node.State.Definition.Id then
                        writerAcceptsType node
                    elif nodeHasTypeParameter node then
                        nodeCanUseType node
                        && (drawing.Connectors
                            |> Seq.choose (fun connector ->
                                match connector.Start, connector.End with
                                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                                    when Object.ReferenceEquals(startPin.Parent, node)
                                         && startPin.Kind = DataOutput
                                         && endPin.Kind = DataInput ->
                                    Some endPin
                                | _ ->
                                    None)
                            |> Seq.forall (fun endPin ->
                                match endPin.Parent with
                                | :? PipelineNodeViewModel as downstreamNode when downstreamNode.State.Definition.Id = "Cast" ->
                                    true
                                | :? PipelineNodeViewModel as downstreamNode when nodeHasTypeParameter downstreamNode ->
                                    branchCanAcceptType downstreamNode visited
                                | :? PipelineNodeViewModel as downstreamNode when SourceImageNode.hasOutputTitle downstreamNode.State.Definition.Id ->
                                    writerAcceptsType downstreamNode
                                | _ ->
                                    fixedInputAcceptsType endPin))
                    else
                        true

                let trySetParameter key (node: PipelineNodeViewModel) =
                    node.State.Parameters
                    |> Seq.tryFind (fun parameter -> parameter.Key = key)
                    |> Option.filter parameterCanUseValue
                    |> Option.map (fun parameter ->
                        let changed = parameter.Value <> typeValue
                        if changed then
                            parameter.Value <- typeValue

                        changed)
                    |> Option.defaultValue false

                let changedNodes = ResizeArray<PipelineNodeViewModel>()
                let queue = Queue<PipelineNodeViewModel>()
                let visited = HashSet<PipelineNodeViewModel>(HashIdentity.Reference)

                let remember node =
                    if changedNodes |> Seq.exists (fun candidate -> Object.ReferenceEquals(candidate, node)) |> not then
                        changedNodes.Add node

                remember sourceNode
                queue.Enqueue sourceNode

                while queue.Count > 0 do
                    let node = queue.Dequeue()

                    if visited.Add node then
                        let downstreamInputs =
                            drawing.Connectors
                            |> Seq.choose (fun connector ->
                                match connector.Start, connector.End with
                                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                                    when Object.ReferenceEquals(startPin.Parent, node)
                                         && startPin.Kind = DataOutput
                                         && endPin.Kind = DataInput ->
                                    Some endPin
                                | _ ->
                                    None)
                            |> Seq.toArray

                        for endPin in downstreamInputs do
                            match endPin.Parent with
                            | :? PipelineNodeViewModel as targetNode when targetNode.State.Definition.Id = "Cast" ->
                                if trySetParameter "sourceType" targetNode then
                                    remember targetNode
                            | :? PipelineNodeViewModel as targetNode ->
                                if nodeHasTypeParameter targetNode then
                                    let branchIsCompatible = branchCanAcceptType targetNode (HashSet<PipelineNodeViewModel>(HashIdentity.Reference))

                                    if branchIsCompatible then
                                        if trySetParameter "type" targetNode then
                                            remember targetNode
                                            queue.Enqueue targetNode
                                    else
                                        drawing.Connectors
                                        |> Seq.tryFind (fun connector -> Object.ReferenceEquals(connector.End, endPin))
                                        |> Option.iter (fun connector -> drawing.Connectors.Remove(connector) |> ignore)
                                elif not (fixedInputAcceptsType endPin) then
                                    drawing.Connectors
                                    |> Seq.tryFind (fun connector -> Object.ReferenceEquals(connector.End, endPin))
                                    |> Option.iter (fun connector -> drawing.Connectors.Remove(connector) |> ignore)
                                else
                                    match endPin.Parent with
                                    | :? PipelineNodeViewModel as fixedNode when SourceImageNode.hasOutputTitle fixedNode.State.Definition.Id ->
                                        remember fixedNode
                                    | _ ->
                                        ()
                            | _ ->
                                ()

                let refreshChangedNodePinsTogether (nodes: PipelineNodeViewModel seq) =
                    let nodes = nodes |> Seq.toArray
                    let nodeSet = HashSet<PipelineNodeViewModel>(nodes, HashIdentity.Reference)

                    let dynamicPinIndexFor (node: PipelineNodeViewModel) (pin: PipelinePinViewModel) =
                        match pin.Kind with
                        | ParameterInput ->
                            node.State.Parameters
                            |> Seq.tryFindIndex (fun parameter -> parameter.Key = pin.ParameterKey)
                        | DataInput
                        | DataOutput ->
                            match pin.Kind, pin.Port.Name with
                            | DataInput, name when name.StartsWith("J:", StringComparison.Ordinal) -> Some 1
                            | DataInput, name when name.StartsWith("K:", StringComparison.Ordinal) -> Some 2
                            | DataInput, name when name.StartsWith("L:", StringComparison.Ordinal) -> Some 3
                            | _ -> Some 0
                        | ScalarOutput ->
                            Some 0
                        | ReducerOutput ->
                            node.Pins
                            |> Seq.choose (function
                                | :? PipelinePinViewModel as outputPin when outputPin.Kind = ReducerOutput -> Some outputPin
                                | _ -> None)
                            |> Seq.tryFindIndex (fun outputPin -> outputPin.Port.Name = pin.Port.Name)

                    let nodeEndpoint (pin: IPin) =
                        match pin with
                        | :? PipelinePinViewModel as pipelinePin ->
                            match pipelinePin.Parent with
                            | :? PipelineNodeViewModel as node when nodeSet.Contains node ->
                                true, Some node, dynamicPinIndexFor node pipelinePin |> Option.map (fun index -> pipelinePin.Kind, index)
                            | _ ->
                                false, None, None
                        | _ ->
                            false, None, None

                    let connectors =
                        drawing.Connectors
                        |> Seq.filter (fun connector ->
                            match connector.Start.Parent, connector.End.Parent with
                            | (:? PipelineNodeViewModel as startNode), _ when nodeSet.Contains startNode -> true
                            | _, (:? PipelineNodeViewModel as endNode) when nodeSet.Contains endNode -> true
                            | _ -> false)
                        |> Seq.toArray

                    let connectorSnapshots =
                        connectors
                        |> Array.map (fun connector ->
                            let startBelongsToNode, startNode, startEndpoint = nodeEndpoint connector.Start
                            let endBelongsToNode, endNode, endEndpoint = nodeEndpoint connector.End
                            connector, startBelongsToNode, startNode, startEndpoint, endBelongsToNode, endNode, endEndpoint)

                    for connector in connectors do
                        drawing.Connectors.Remove(connector) |> ignore

                    for connector, startBelongsToNode, startNode, startEndpoint, endBelongsToNode, endNode, endEndpoint in connectorSnapshots do
                        let startReplacement =
                            match startNode, startEndpoint with
                            | Some node, Some(kind, index) -> pinByKindIndex kind index node
                            | _ -> None

                        let endReplacement =
                            match endNode, endEndpoint with
                            | Some node, Some(kind, index) -> pinByKindIndex kind index node
                            | _ -> None

                        let startStillExists =
                            if startBelongsToNode then startReplacement.IsSome else true

                        let endStillExists =
                            if endBelongsToNode then endReplacement.IsSome else true

                        startReplacement
                        |> Option.iter (fun pin -> connector.Start <- pin)

                        endReplacement
                        |> Option.iter (fun pin -> connector.End <- pin)

                        if startStillExists && endStillExists && canConnectPins connector.Start connector.End then
                            drawing.Connectors.Add(connector) |> ignore

                for node in changedNodes do
                    node.RebuildPins()

                refreshChangedNodePinsTogether changedNodes

                true
            finally
                suppressTypePropagation <- false
                suppressConnectorCollectionRefresh <- false

    let createNode index functionId =
        let node =
            PipelineNodeViewModel(
                createState functionId,
                (fun node -> this.SelectNodeFromEditor node),
                moveSelectedNodesBy,
                (fun () -> drawing.Width, drawing.Height),
                (fun () -> this.MarkGraphDirty()),
                removePinConnections,
                refreshNodePins,
                handleTypeChange)

        watchState node.State

        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.ClampToDrawing()
        node.SyncMoveOrigin()
        node

    let addConnector (startPin: IPin) (endPin: IPin) =
        let connector = ConnectorViewModel()
        connector.Start <- startPin
        connector.End <- endPin
        connector.Orientation <- connectorOrientation startPin endPin
        drawing.Connectors.Add(connector :> IConnector)

        match endPin with
        | :? PipelinePinViewModel as inputPin ->
            match inputPin.Parent with
            | :? PipelineNodeViewModel as node when node.State.Definition.Id = "Expand" ->
                match startPin with
                | :? PipelinePinViewModel as outputPin ->
                    node.State.RecordType <- Some outputPin.Port.Type
                    node.RebuildPins()
                    refreshNodePins node
                | _ -> ()
            | _ -> ()
        | _ -> ()

        this.MarkGraphDirty()

    let refreshExpandNodeRecordTypes () =
        if not suppressExpandRefresh then
            suppressExpandRefresh <- true

            try
                let pinIsStillOnNode (pin: IPin) =
                    match pin with
                    | :? PipelinePinViewModel as pipelinePin ->
                        match pipelinePin.Parent with
                        | :? PipelineNodeViewModel as node ->
                            node.Pins
                            |> Seq.exists (fun currentPin -> Object.ReferenceEquals(currentPin, pin))
                        | _ -> true
                    | _ -> true

                let pruneInvalidConnectors () =
                    let invalidConnectors =
                        drawing.Connectors
                        |> Seq.filter (fun connector ->
                            not (pinIsStillOnNode connector.Start)
                            || not (pinIsStillOnNode connector.End)
                            || not (canConnectPins connector.Start connector.End))
                        |> Seq.toArray

                    for connector in invalidConnectors do
                        drawing.Connectors.Remove(connector) |> ignore

                    invalidConnectors.Length > 0

                let refreshExpandNodes () =
                    let mutable changed = false

                    drawing.Nodes
                    |> Seq.choose (function
                        | :? PipelineNodeViewModel as node when node.State.Definition.Id = "Expand" -> Some node
                        | _ -> None)
                    |> Seq.iter (fun node ->
                        let recordType =
                            drawing.Connectors
                            |> Seq.tryPick (fun connector ->
                                match connector.Start, connector.End with
                                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin)
                                    when Object.ReferenceEquals(endPin.Parent, node)
                                         && endPin.Kind = DataInput ->
                                    Some startPin.Port.Type
                                | _ -> None)

                        if node.State.RecordType <> recordType then
                            node.State.RecordType <- recordType
                            node.RebuildPins()
                            refreshNodePins node
                            changed <- true)

                    changed

                let mutable changed = true
                let mutable passes = 0

                while changed && passes < 32 do
                    passes <- passes + 1
                    // Walk the graph forward: prune invalid outgoing links first, then let
                    // dynamic downstream nodes adapt and possibly invalidate their outputs.
                    changed <- pruneInvalidConnectors() || refreshExpandNodes()
            finally
                suppressExpandRefresh <- false

    let scheduleExpandNodeRecordTypeRefresh () =
        if not suppressExpandRefresh && not expandRefreshScheduled then
            expandRefreshScheduled <- true
            Dispatcher.UIThread.Post(
                (fun () ->
                    expandRefreshScheduled <- false
                    refreshExpandNodeRecordTypes()),
                DispatcherPriority.Background)

    do
        updatePaletteGroups()

        match drawing.Nodes with
        | :? INotifyCollectionChanged as nodes ->
            nodes.CollectionChanged.Add(fun _ ->
                this.RaiseGraphStateChanged()
                this.MarkGraphDirty())
        | _ -> ()

        match drawing.Connectors with
        | :? INotifyCollectionChanged as connectors ->
            connectors.CollectionChanged.Add(fun _ ->
                if not suppressConnectorCollectionRefresh then
                    scheduleExpandNodeRecordTypeRefresh()
                    refreshScalarTypeOptions()
                    refreshScalarOpTypeOptions()
                    refreshConnectedImageInputPins()
                    refreshImageFormatOptions()
                    refreshSourceImageTypeOptions()
                    refreshCastTypeOptions()
                    refreshPairOperationTypeOptions()
                    refreshScalarImageOperationTypeOptions()
                    refreshThresholdTypeOptions()
                    refreshSerialTransformTypeOptions()
                    this.RaiseGraphStateChanged()
                    this.MarkGraphDirty())
        | _ -> ()

        //addSeedNodes()

    member _.Editor = editor
    member _.PaletteGroups = paletteGroups
    member _.ArrangeSettings = arrangeSettings

    member this.PaletteSearch
        with get () = paletteSearch
        and set value =
            if this.SetProperty(&paletteSearch, value) then
                updatePaletteGroups()

    member this.SelectedNode
        with get () = selectedNode
        and set value =
            if this.SetProperty(&selectedNode, value) then
                selectOnlyNode value
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))

    member this.SelectedElement
        with get () = selectedNode
        and set value = this.SelectedNode <- value

    member _.HasSelectedElement = not (isNull selectedNode)

    member this.ClearSelection() =
        if selectedNodes.Count > 0 || not (isNull selectedNode) then
            clearSelectedNodes()
            selectedNode <- null
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))

    member this.SelectSingleNode(node: PipelineNodeViewModel) =
        this.SelectedNode <- node

    member this.SelectNodeFromEditor(node: PipelineNodeViewModel) =
        if not (isNull node) && node.State.IsSelected && selectedNodes.Count > 1 then
            selectedNode <- node
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))
        else
            this.SelectedNode <- node

    member this.ToggleNodeSelection(node: PipelineNodeViewModel) =
        if not (isNull node) then
            if selectedNodes.Contains node then
                selectedNodes.Remove node |> ignore
                node.State.IsSelected <- false

                if Object.ReferenceEquals(selectedNode, node) then
                    selectedNode <-
                        selectedNodes
                        |> Seq.tryHead
                        |> Option.toObj

                    this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
                    this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
                    this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
                    this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))
            else
                addSelectedNode node
                selectedNode <- node
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))

    member this.SelectNodes(nodes: PipelineNodeViewModel seq) =
        clearSelectedNodes()

        let selected = nodes |> Seq.toArray

        selected |> Array.iter addSelectedNode

        selectedNode <-
            selected
            |> Array.tryLast
            |> Option.toObj

        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedNodes))

    member this.MoveSelectionBy(dx: float, dy: float) =
        let selected = selectedNodes |> Seq.toArray

        if selected.Length > 0 && (dx <> 0. || dy <> 0.) then
            let clampedDx, clampedDy = clampSelectionDelta selected dx dy
            applySelectionDelta selected clampedDx clampedDy

    member _.HasSelectedNodes = selectedNodes.Count > 0 || not (isNull selectedNode)

    member _.SelectedNodes =
        if selectedNodes.Count > 0 then
            selectedNodes |> Seq.toArray
        elif not (isNull selectedNode) then
            [| selectedNode |]
        else
            Array.empty

    member _.GraphOutput = graphOutput

    member _.GeneratedProgram = graphOutput

    member private this.SetGraphOutput(text: string) =
        graphOutput <- text
        this.RaiseGraphOutputChanged()

    member this.AppendGraphOutput(text: string) =
        let separator =
            if String.IsNullOrEmpty graphOutput
               || graphOutput.EndsWith(Environment.NewLine, StringComparison.Ordinal)
               || text.StartsWith(Environment.NewLine, StringComparison.Ordinal) then
                ""
            else
                Environment.NewLine

        graphOutput <- graphOutput + separator + text
        this.RaiseGraphOutputChanged()

    member this.AppendGraphOutputLine(text: string) =
        this.AppendGraphOutput(text + Environment.NewLine)

    member private this.AppendGeneratedProgram(text: string) =
        let timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
        let block = $"Program generated {timestamp}{Environment.NewLine}{text}"
        this.AppendGraphOutput(block)

    member private this.FlushPendingGraphOutput() =
        let text =
            lock pendingGraphOutputLock (fun () ->
                let text = pendingGraphOutput.ToString()
                pendingGraphOutput.Clear() |> ignore
                graphOutputFlushScheduled <- false
                text)

        if not (String.IsNullOrEmpty text) then
            this.AppendGraphOutput(text)

    member private this.AppendGraphOutputOnUi(text: string) =
        let scheduleFlush =
            lock pendingGraphOutputLock (fun () ->
                pendingGraphOutput.Append(text) |> ignore

                if graphOutputFlushScheduled then
                    false
                else
                    graphOutputFlushScheduled <- true
                    true)

        if scheduleFlush then
            Dispatcher.UIThread.Post((fun () -> this.FlushPendingGraphOutput()), DispatcherPriority.Background)

    member private this.AppendGraphOutputLineOnUi(text: string) =
        this.AppendGraphOutputOnUi(text + Environment.NewLine)

    member private this.SetRunInProgress(value: bool) =
        if Dispatcher.UIThread.CheckAccess() then
            if isRunInProgress <> value then
                isRunInProgress <- value
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanRunGraph))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanStopRun))
        else
            Dispatcher.UIThread.Post(fun () -> this.SetRunInProgress(value))

    member private _.SetActiveRunProcess(procOpt: Process option) =
        lock activeRunProcessLock (fun () -> activeRunProcess <- procOpt)

    member private _.KillActiveRunProcess() =
        lock activeRunProcessLock (fun () ->
            match activeRunProcess with
            | Some proc ->
                try
                    if not proc.HasExited then
                        proc.Kill(true)
                with _ ->
                    ()
            | None -> ())

    member private _.FindRepositoryRoot() =
        let rec search (directory: DirectoryInfo) =
            let stackProcessingProject =
                Path.Combine(directory.FullName, "src", "StackProcessing", "StackProcessing.fsproj")

            if File.Exists stackProcessingProject then
                Some directory.FullName
            elif isNull directory.Parent then
                None
            else
                search directory.Parent

        let start =
            DirectoryInfo(AppContext.BaseDirectory)

        search start
        |> Option.defaultWith (fun () -> Directory.GetCurrentDirectory())

    member private this.DotnetExecutable() =
        let macInstall = "/usr/local/share/dotnet/dotnet"

        if File.Exists macInstall then
            macInstall
        else
            "dotnet"

    member private _.GraphRunWorkingDirectory() =
        currentGraphPath
        |> Option.bind (fun path ->
            let fullPath = Path.GetFullPath(path)
            let directory = Path.GetDirectoryName(fullPath)

            if String.IsNullOrWhiteSpace directory then
                None
            else
                Some directory)
        |> Option.defaultWith (fun () ->
            let home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)

            if String.IsNullOrWhiteSpace home then
                studioWorkingDirectory
            else
                home)

    member private this.EnsureRunProject(includePlotly: bool) =
        Directory.CreateDirectory(runProjectDirectory) |> ignore

        let repositoryRoot = this.FindRepositoryRoot()
        let stackProcessingProject = Path.Combine(repositoryRoot, "src", "StackProcessing", "StackProcessing.fsproj")
        let stackProcessingCoreProject = Path.Combine(repositoryRoot, "src", "StackProcessing.Core", "StackProcessing.Core.fsproj")
        let simpleItkManaged = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpManaged.dll")
        let simpleItkWindowsNative = Path.Combine(repositoryRoot, "lib", "SimpleITKCSharpNative.dll")
        let simpleItkLinuxNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.so")
        let simpleItkMacNative = Path.Combine(repositoryRoot, "lib", "libSimpleITKCSharpNative.dylib")
        let projectPath = Path.Combine(runProjectDirectory, "StudioRun.fsproj")

        let xml value =
            SecurityElement.Escape(value)

        let plotlyReference =
            if includePlotly then
                """    <PackageReference Include="Plotly.NET" Version="5.1.0" />
"""
            else
                ""

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
{plotlyReference}
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

        if not (File.Exists projectPath) || File.ReadAllText(projectPath) <> projectFile then
            File.WriteAllText(projectPath, projectFile)

        projectPath

    member private this.WriteRunProgram(generatedProgram: string) =
        let programPath = Path.Combine(runProjectDirectory, "Program.fs")
        File.WriteAllText(programPath, generatedProgram, Encoding.UTF8)

    member private this.RunProcess(phase: string option) (echoOutput: bool) (echoOnlyOnFailure: bool) (fileName: string) (arguments: string list) (workingDirectory: string) (cancellationToken: CancellationToken) =
        async {
            phase |> Option.iter this.AppendGraphOutputLineOnUi

            let startInfo = ProcessStartInfo()
            startInfo.FileName <- fileName
            startInfo.WorkingDirectory <- workingDirectory
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true

            arguments
            |> List.iter startInfo.ArgumentList.Add

            use proc = new Process()
            proc.StartInfo <- startInfo
            proc.EnableRaisingEvents <- true

            let readLines (reader: StreamReader) =
                async {
                    let lines = ResizeArray<string>()
                    let mutable keepReading = true

                    while keepReading do
                        let! line = reader.ReadLineAsync() |> Async.AwaitTask

                        if isNull line then
                            keepReading <- false
                        else
                            if echoOutput && echoOnlyOnFailure then
                                lines.Add line

                            if echoOutput && not echoOnlyOnFailure then
                                this.AppendGraphOutputLineOnUi(line)

                    return lines |> Seq.toList
                }

            if not (proc.Start()) then
                return -1
            else
                this.SetActiveRunProcess(Some proc)
                use _registration =
                    cancellationToken.Register(fun () ->
                        try
                            if not proc.HasExited then
                                proc.Kill(true)
                        with _ ->
                            ())

                try
                    let! output = readLines proc.StandardOutput |> Async.StartChild
                    let! error = readLines proc.StandardError |> Async.StartChild

                    try
                        do! proc.WaitForExitAsync(cancellationToken) |> Async.AwaitTask
                    with
                    | :? OperationCanceledException ->
                        if not proc.HasExited then
                            try proc.Kill(true) with _ -> ()
                        do! proc.WaitForExitAsync() |> Async.AwaitTask

                    let! outputLines = output
                    let! errorLines = error

                    if echoOutput && echoOnlyOnFailure && proc.ExitCode <> 0 && not cancellationToken.IsCancellationRequested then
                        (outputLines @ errorLines)
                        |> List.iter this.AppendGraphOutputLineOnUi

                    return proc.ExitCode
                finally
                    this.SetActiveRunProcess(None)
        }

    member private this.BuildAndRunGeneratedProgram(generatedProgram: string) (cancellationToken: CancellationToken) =
        async {
            try
                this.SetRunInProgress(true)
                this.AppendGraphOutputLineOnUi("Compiling")

                let projectPath = this.EnsureRunProject(generatedProgram.Contains("open Plotly.NET", StringComparison.Ordinal))
                this.WriteRunProgram(generatedProgram)

                let dotnet = this.DotnetExecutable()

                let! buildExitCode =
                    this.RunProcess
                        None
                        true
                        true
                        dotnet
                        [ "build"; projectPath; "--configuration"; "Release"; "--nologo"; "--verbosity"; "quiet"; "--consoleLoggerParameters:ErrorsOnly" ]
                        runProjectDirectory
                        cancellationToken

                if cancellationToken.IsCancellationRequested then
                    this.AppendGraphOutputLineOnUi("Run stopped")
                elif buildExitCode = 0 then
                    let runWorkingDirectory = this.GraphRunWorkingDirectory()
                    let runStarted = DateTimeOffset.Now

                    let! runExitCode =
                        this.RunProcess
                            (Some "Running")
                            true
                            false
                            dotnet
                            [ "run"; "--configuration"; "Release"; "--no-build"; "--project"; projectPath ]
                            runWorkingDirectory
                            cancellationToken

                    let elapsed = DateTimeOffset.Now - runStarted

                    if cancellationToken.IsCancellationRequested then
                        this.AppendGraphOutputLineOnUi($"Run stopped after {elapsed:g}")
                    elif runExitCode = 0 then
                        this.AppendGraphOutputLineOnUi($"Run completed in {elapsed:g}")
                    else
                        this.AppendGraphOutputLineOnUi($"Run failed with exit code {runExitCode} after {elapsed:g}")
                else
                    this.AppendGraphOutputLineOnUi($"Build failed with exit code {buildExitCode}")
            with ex ->
                this.AppendGraphOutputLineOnUi($"Run setup failed: {ex.Message}")

            this.SetRunInProgress(false)
        }

    member this.ResolveFileDirectoryPath
        with get () = resolveFileDirectoryPath
        and set value = resolveFileDirectoryPath <- value

    member private _.SetFileDirectoryPromptHighlight(node: PipelineNodeViewModel, isHighlighted: bool) =
        let apply () =
            if isHighlighted then
                selectOnlyNode node

            node.State.IsProblemHighlighted <- isHighlighted

        if Dispatcher.UIThread.CheckAccess() then
            apply()
        else
            Dispatcher.UIThread.Post(apply)

    member private this.SetFileDirectoryValue(node: PipelineNodeViewModel, path: string) =
        let apply () =
            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "value")
            |> Option.iter (fun parameter -> parameter.Value <- path)

            node.State.Title <- FileDirectoryNode.title node.State
            node.Name <- node.State.Title
            this.MarkGraphDirty()

        if Dispatcher.UIThread.CheckAccess() then
            apply()
        else
            Dispatcher.UIThread.Post(apply)

    member private this.ResolveFileDirectoryInputs() =
        async {
            let mutable shouldContinue = true

            for node in fileDirectoryNodes () do
                if shouldContinue then
                    this.SetFileDirectoryPromptHighlight(node, true)

                    let! selectedPath = resolveFileDirectoryPath node

                    this.SetFileDirectoryPromptHighlight(node, false)

                    match selectedPath with
                    | Some path when not (String.IsNullOrWhiteSpace path) ->
                        this.SetFileDirectoryValue(node, path)
                    | _ ->
                        shouldContinue <- false
                        this.AppendGraphOutputLineOnUi("Run cancelled: file/directory selection was not completed.")

            return shouldContinue
        }
    
    member _.ConnectSeedPipeline() =
        if drawing.Connectors.Count = 0 then
            pipelineNodes ()
            |> Seq.pairwise
            |> Seq.iter (fun (left, right) ->
                match tryPin PinAlignment.Right left, tryPin PinAlignment.Left right with
                | Some startPin, Some endPin when canConnectPins startPin endPin ->
                    addConnector startPin endPin
                | _ -> ())

    member this.AddReadCommand =
        SimpleCommand((fun _ -> this.AddElement("Read")), (fun _ -> true)) :> ICommand

    member this.AddGaussianCommand =
        SimpleCommand((fun _ -> this.AddElement("SmoothWGauss")), (fun _ -> true)) :> ICommand

    member this.AddCastCommand =
        SimpleCommand((fun _ -> this.AddElement("Cast")), (fun _ -> true)) :> ICommand

    member this.AddWriteCommand =
        SimpleCommand((fun _ -> this.AddElement("Write")), (fun _ -> true)) :> ICommand

    member this.AddPaletteElementCommand =
        SimpleCommand(
            (fun parameter ->
                match parameter with
                | :? Function as definition -> this.AddElement(definition.Id)
                | :? string as functionId -> this.AddElement(functionId)
                | _ -> ()),
            (fun _ -> true))
        :> ICommand

    member this.ArrangeGraph() =
        arrangeRunSerial <- arrangeRunSerial + 1
        let runSerial = arrangeRunSerial

        async {
            let nodes = pipelineNodes () |> Seq.toArray

            if nodes.Length > 0 then
                let connectorEdges =
                    drawing.Connectors
                    |> Seq.choose (fun connector ->
                        match connector.Start, connector.End with
                        | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                            match startPin.Parent, endPin.Parent with
                            | (:? PipelineNodeViewModel as startNode), (:? PipelineNodeViewModel as endNode)
                                when not (Object.ReferenceEquals(startNode, endNode)) ->
                                Some(startPin, endPin, startNode, endNode)
                            | _ -> None
                        | _ -> None)
                    |> Seq.toArray

                let arrangeWeights =
                    {| BoundingBoxCompact = arrangeSettings.BoundingBoxCompact
                       BoundingBoxSquare = arrangeSettings.BoundingBoxSquare
                       BoxSpacing = arrangeSettings.BoxSpacing
                       BoxSpacingSharpness = arrangeSettings.BoxSpacingSharpness
                       ConnectorDirection = arrangeSettings.ConnectorDirection
                       ConnectorLength = arrangeSettings.ConnectorLength
                       ConnectorAvoidsBoxes = arrangeSettings.ConnectorAvoidsBoxes
                       ConnectorCrossings = arrangeSettings.ConnectorCrossings
                       StayNearUserLayout = arrangeSettings.StayNearUserLayout
                       Damping = arrangeSettings.Damping
                       VisualizationDelayMs = int arrangeSettings.VisualizationDelayMs |}

                let arrangeTiming =
                    {| Iterations = 1000
                       VisualizeEvery = 3
                       StopWhenMaxMoveBelow = 0.5
                       InitialMaxMove = 8.0
                       FinalMaxMove = 1.0 |}

                let leftMargin = 32.
                let topMargin = 32.

                let connectorAxis (startPin: PipelinePinViewModel) (endPin: PipelinePinViewModel) =
                    match startPin.Alignment, endPin.Alignment with
                    | PinAlignment.Right, PinAlignment.Left -> Some "horizontal"
                    | PinAlignment.Bottom, PinAlignment.Top -> Some "vertical"
                    | _ -> None

                let nodeIndexes = Dictionary<PipelineNodeViewModel, int>(HashIdentity.Reference)
                nodes |> Array.iteri (fun index node -> nodeIndexes[node] <- index)

                let connectedNodePairs = HashSet<int * int>()

                connectorEdges
                |> Array.iter (fun (_, _, startNode, endNode) ->
                    match nodeIndexes.TryGetValue startNode, nodeIndexes.TryGetValue endNode with
                    | (true, startIndex), (true, endIndex) ->
                        let pair = if startIndex < endIndex then startIndex, endIndex else endIndex, startIndex
                        connectedNodePairs.Add(pair) |> ignore
                    | _ -> ())

                let verticalFanoutOffsets = Dictionary<int, float>()
                let horizontalFanoutOffsets = Dictionary<int, float>()

                connectorEdges
                |> Array.indexed
                |> Array.choose (fun (edgeIndex, (startPin, endPin, startNode, endNode)) ->
                    match connectorAxis startPin endPin, nodeIndexes.TryGetValue startNode, nodeIndexes.TryGetValue endNode with
                    | Some "vertical", (true, startIndex), (true, endIndex) ->
                        Some(edgeIndex, startPin, startIndex, endIndex)
                    | _ ->
                        None)
                |> Seq.groupBy (fun (_, startPin, _, _) -> startPin)
                |> Seq.iter (fun (_, edges) ->
                    let edges =
                        edges
                        |> Seq.sortBy (fun (_, _, _, endIndex) -> nodes[endIndex].X + nodes[endIndex].Width / 2.)
                        |> Seq.toArray

                    if edges.Length > 1 then
                        let positions = Array.zeroCreate<float> edges.Length

                        for i in 1 .. edges.Length - 1 do
                            let _, _, _, previousEnd = edges[i - 1]
                            let _, _, _, currentEnd = edges[i]
                            let siblingGap = max nodes[previousEnd].Height nodes[currentEnd].Height
                            let centerSpacing =
                                nodes[previousEnd].Width / 2.
                                + nodes[currentEnd].Width / 2.
                                + siblingGap

                            positions[i] <- positions[i - 1] + centerSpacing

                        let center = (positions[0] + positions[positions.Length - 1]) / 2.

                        edges
                        |> Array.iteri (fun i (edgeIndex, _, _, _) ->
                            verticalFanoutOffsets[edgeIndex] <- positions[i] - center))

                connectorEdges
                |> Array.indexed
                |> Array.choose (fun (edgeIndex, (startPin, endPin, startNode, endNode)) ->
                    match connectorAxis startPin endPin, nodeIndexes.TryGetValue startNode, nodeIndexes.TryGetValue endNode with
                    | Some "horizontal", (true, startIndex), (true, endIndex) ->
                        Some(edgeIndex, endPin, startIndex, endIndex)
                    | _ ->
                        None)
                |> Seq.groupBy (fun (_, _, _, endIndex) -> endIndex)
                |> Seq.iter (fun (_, edges) ->
                    let edges =
                        edges
                        |> Seq.sortBy (fun (_, endPin, _, _) -> endPin.Y)
                        |> Seq.toArray

                    if edges.Length > 1 then
                        let positions = Array.zeroCreate<float> edges.Length

                        for i in 1 .. edges.Length - 1 do
                            let _, _, previousStart, _ = edges[i - 1]
                            let _, _, currentStart, _ = edges[i]
                            let siblingGap = max nodes[previousStart].Height nodes[currentStart].Height
                            let centerSpacing =
                                nodes[previousStart].Height / 2.
                                + nodes[currentStart].Height / 2.
                                + siblingGap

                            positions[i] <- positions[i - 1] + centerSpacing

                        let spreadCenter = (positions[0] + positions[positions.Length - 1]) / 2.
                        let targetCenter = edges |> Array.averageBy (fun (_, endPin, _, _) -> endPin.Y)

                        edges
                        |> Array.iteri (fun i (edgeIndex, endPin, _, _) ->
                            horizontalFanoutOffsets[edgeIndex] <- endPin.Y - targetCenter - (positions[i] - spreadCenter)))

                let startX = nodes |> Array.map _.X
                let startY = nodes |> Array.map _.Y
                let x = Array.copy startX
                let y = Array.copy startY
                let fx = Array.zeroCreate<float> nodes.Length
                let fy = Array.zeroCreate<float> nodes.Length
                let vx = Array.zeroCreate<float> nodes.Length
                let vy = Array.zeroCreate<float> nodes.Length

                let inline clamp lo hi value = min hi (max lo value)
                let inline signOrOne value = if value < 0. then -1. else 1.
                let inline square value = value * value

                let inline nodeCenterX i = x[i] + nodes[i].Width / 2.
                let inline nodeCenterY i = y[i] + nodes[i].Height / 2.
                let inline rectLeft i = x[i]
                let inline rectTop i = y[i]
                let inline rectRight i = x[i] + nodes[i].Width
                let inline rectBottom i = y[i] + nodes[i].Height

                let addForce index dx dy scale =
                    fx[index] <- fx[index] + dx * scale
                    fy[index] <- fy[index] + dy * scale

                let addPairForce first second dx dy scale =
                    addForce first -dx -dy scale
                    addForce second dx dy scale

                let applyPositions () =
                    for i in 0 .. nodes.Length - 1 do
                        nodes[i].X <- x[i]
                        nodes[i].Y <- y[i]

                let pinPoint (positionsX: float array) (positionsY: float array) index (pin: PipelinePinViewModel) =
                    positionsX[index] + pin.X, positionsY[index] + pin.Y

                let connectorSegments () =
                    connectorEdges
                    |> Array.choose (fun (startPin, endPin, startNode, endNode) ->
                        match nodeIndexes.TryGetValue startNode, nodeIndexes.TryGetValue endNode with
                        | (true, startIndex), (true, endIndex) ->
                            let sx, sy = pinPoint x y startIndex startPin
                            let ex, ey = pinPoint x y endIndex endPin
                            Some(startIndex, endIndex, sx, sy, ex, ey)
                        | _ -> None)

                let distanceFromPointToSegment px py ax ay bx by =
                    let vx = bx - ax
                    let vy = by - ay
                    let lengthSquared = vx * vx + vy * vy

                    let t =
                        if lengthSquared <= 0.000001 then
                            0.
                        else
                            ((px - ax) * vx + (py - ay) * vy) / lengthSquared
                            |> clamp 0. 1.

                    let cx = ax + t * vx
                    let cy = ay + t * vy
                    let dx = px - cx
                    let dy = py - cy
                    sqrt (dx * dx + dy * dy), dx, dy

                let ccw ax ay bx by cx cy =
                    (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

                let segmentsCross ax ay bx by cx cy dx dy =
                    ccw ax ay cx cy dx dy <> ccw bx by cx cy dx dy
                    && ccw ax ay bx by cx cy <> ccw ax ay bx by dx dy

                let mutable iteration = 0
                let mutable isConverged = false

                while iteration < arrangeTiming.Iterations && not isConverged do
                    Array.Clear(fx, 0, fx.Length)
                    Array.Clear(fy, 0, fy.Length)

                    let minX = nodes |> Array.mapi (fun i _ -> rectLeft i) |> Array.min
                    let minY = nodes |> Array.mapi (fun i _ -> rectTop i) |> Array.min
                    let maxX = nodes |> Array.mapi (fun i _ -> rectRight i) |> Array.max
                    let maxY = nodes |> Array.mapi (fun i _ -> rectBottom i) |> Array.max
                    let width = max 1. (maxX - minX)
                    let height = max 1. (maxY - minY)
                    let centerX = minX + width / 2.
                    let centerY = minY + height / 2.
                    let squareBias = width - height
                    let boundaryTolerance = 0.5
                    let compactPerimeterScale =
                        arrangeWeights.BoundingBoxCompact * 0.5 * (width + height)
                    let compactCenterScale =
                        compactPerimeterScale / max width height

                    for i in 0 .. nodes.Length - 1 do
                        let cx = nodeCenterX i
                        let cy = nodeCenterY i

                        if abs (rectLeft i - minX) <= boundaryTolerance then
                            addForce i 1. 0. compactPerimeterScale

                        if abs (rectRight i - maxX) <= boundaryTolerance then
                            addForce i -1. 0. compactPerimeterScale

                        if abs (rectTop i - minY) <= boundaryTolerance then
                            addForce i 0. 1. compactPerimeterScale

                        if abs (rectBottom i - maxY) <= boundaryTolerance then
                            addForce i 0. -1. compactPerimeterScale

                        // Bounding-box compactness uses perimeter, not area. Area rewards collapsing
                        // one dimension when the graph is already elongated; perimeter keeps the
                        // inward pressure symmetric in x and y.
                        //
                        // The exact bounding-box gradient only touches nodes on the current
                        // min/max borders. This soft companion keeps interior nodes from drifting
                        // far apart when another node temporarily owns the graph boundary.
                        addForce i (centerX - cx) 0. compactCenterScale
                        addForce i 0. (centerY - cy) compactCenterScale
                        addForce i (startX[i] - x[i]) (startY[i] - y[i]) arrangeWeights.StayNearUserLayout
                        addForce i (centerX - cx) 0. (arrangeWeights.BoundingBoxSquare * squareBias / width)
                        addForce i 0. (centerY - cy) (arrangeWeights.BoundingBoxSquare * -squareBias / height)

                    for i in 0 .. nodes.Length - 2 do
                        for j in i + 1 .. nodes.Length - 1 do
                            let isConnectedPair = connectedNodePairs.Contains(i, j)
                            let dx = nodeCenterX j - nodeCenterX i
                            let dy = nodeCenterY j - nodeCenterY i
                            let horizontalLimit = (nodes[i].Width + nodes[j].Width) / 2.
                            let verticalLimit = (nodes[i].Height + nodes[j].Height) / 2.
                            let standardGap = max nodes[i].Height nodes[j].Height
                            let overlapX = horizontalLimit - abs dx
                            let overlapY = verticalLimit - abs dy

                            let clearance =
                                if overlapX > 0. && overlapY > 0. then
                                    -min overlapX overlapY
                                else
                                    let gapX = max 0. -overlapX
                                    let gapY = max 0. -overlapY
                                    sqrt (square gapX + square gapY)

                            let directionX, directionY =
                                if overlapX > 0. && overlapY > 0. then
                                    if overlapX < overlapY then
                                        signOrOne dx, 0.
                                    else
                                        0., signOrOne dy
                                else
                                    let gapX = max 0. -overlapX
                                    let gapY = max 0. -overlapY

                                    if gapX > 0. && gapY > 0. then
                                        let length = max 0.0001 (sqrt (square gapX + square gapY))
                                        signOrOne dx * gapX / length, signOrOne dy * gapY / length
                                    elif gapX > 0. then
                                        signOrOne dx, 0.
                                    elif gapY > 0. then
                                        0., signOrOne dy
                                    else
                                        0., 0.

                            let normalizedClearance = clearance / max 1. standardGap
                            let spacingForce =
                                arrangeWeights.BoxSpacing
                                * exp (-arrangeWeights.BoxSpacingSharpness * normalizedClearance)

                            if overlapX > 0. && overlapY > 0. then
                                addPairForce i j directionX directionY (spacingForce * (1. + min overlapX overlapY))
                            elif not isConnectedPair && clearance < 3.0 * standardGap then
                                addPairForce i j directionX directionY spacingForce

                    for edgeIndex in 0 .. connectorEdges.Length - 1 do
                        let startPin, endPin, startNode, endNode = connectorEdges[edgeIndex]

                        match nodeIndexes.TryGetValue startNode, nodeIndexes.TryGetValue endNode with
                        | (true, startIndex), (true, endIndex) ->
                            let preferredGap = max nodes[startIndex].Height nodes[endIndex].Height
                            let sx, sy = pinPoint x y startIndex startPin
                            let ex, ey = pinPoint x y endIndex endPin
                            let pinDx = ex - sx
                            let pinDy = ey - sy

                            match connectorAxis startPin endPin with
                            | Some "horizontal" ->
                                let horizontalError = preferredGap - pinDx
                                let desiredDy =
                                    match horizontalFanoutOffsets.TryGetValue edgeIndex with
                                    | true, offset -> offset
                                    | _ -> 0.

                                addPairForce startIndex endIndex 1. 0. (arrangeWeights.ConnectorDirection * horizontalError)
                                addPairForce startIndex endIndex 0. 1. (arrangeWeights.ConnectorDirection * (desiredDy - pinDy))
                            | Some "vertical" ->
                                let verticalError = preferredGap - pinDy
                                let desiredDx =
                                    match verticalFanoutOffsets.TryGetValue edgeIndex with
                                    | true, offset -> offset
                                    | _ -> 0.

                                addPairForce startIndex endIndex 0. 1. (arrangeWeights.ConnectorDirection * verticalError)
                                addPairForce startIndex endIndex 1. 0. (arrangeWeights.ConnectorDirection * (desiredDx - pinDx))
                            | _ -> ()

                            let length = sqrt (square pinDx + square pinDy)
                            let preferredLength = max nodes[startIndex].Height nodes[endIndex].Height
                            let excess = length - preferredLength

                            if excess > 0. && length > 0.0001 then
                                addPairForce startIndex endIndex -(pinDx / length) -(pinDy / length) (arrangeWeights.ConnectorLength * excess)
                            | _ -> ()

                    let segments = connectorSegments()

                    for startIndex, endIndex, sx, sy, ex, ey in segments do
                        for i in 0 .. nodes.Length - 1 do
                            if i <> startIndex && i <> endIndex then
                                let cx = nodeCenterX i
                                let cy = nodeCenterY i
                                let radius = 0.5 * sqrt (square nodes[i].Width + square nodes[i].Height) + 18.
                                let distance, dx, dy = distanceFromPointToSegment cx cy sx sy ex ey

                                if distance < radius then
                                    let nx, ny =
                                        if distance <= 0.0001 then
                                            0., 1.
                                        else
                                            dx / distance, dy / distance

                                    addForce i nx ny (arrangeWeights.ConnectorAvoidsBoxes * (radius - distance))

                    for a in 0 .. segments.Length - 2 do
                        let aStart, aEnd, ax, ay, bx, by = segments[a]

                        for b in a + 1 .. segments.Length - 1 do
                            let bStart, bEnd, cx, cy, dx, dy = segments[b]
                            let sharesNode = aStart = bStart || aStart = bEnd || aEnd = bStart || aEnd = bEnd

                            if not sharesNode && segmentsCross ax ay bx by cx cy dx dy then
                                let anx = -(by - ay)
                                let any = bx - ax
                                let bnx = -(dy - cy)
                                let bny = dx - cx
                                let alen = max 1. (sqrt (square anx + square any))
                                let blen = max 1. (sqrt (square bnx + square bny))

                                addForce aStart (anx / alen) (any / alen) arrangeWeights.ConnectorCrossings
                                addForce aEnd (anx / alen) (any / alen) arrangeWeights.ConnectorCrossings
                                addForce bStart -(bnx / blen) -(bny / blen) arrangeWeights.ConnectorCrossings
                                addForce bEnd -(bnx / blen) -(bny / blen) arrangeWeights.ConnectorCrossings

                    // Most arrange terms are translation-invariant. Remove their shared drift so the
                    // graph settles in place instead of slowly flowing when boundary-node counts differ.
                    let meanForceX = fx |> Array.average
                    let meanForceY = fy |> Array.average

                    for i in 0 .. nodes.Length - 1 do
                        fx[i] <- fx[i] - meanForceX
                        fy[i] <- fy[i] - meanForceY

                    let progress = 1. - float iteration / float arrangeTiming.Iterations
                    let step = 0.75 * progress + 0.08
                    let maxMove = arrangeTiming.InitialMaxMove * progress + arrangeTiming.FinalMaxMove * (1. - progress)
                    let damping = clamp 0. 0.98 arrangeWeights.Damping
                    let mutable maxStepMove = 0.

                    for i in 0 .. nodes.Length - 1 do
                        vx[i] <- damping * vx[i] + (1. - damping) * fx[i] * step
                        vy[i] <- damping * vy[i] + (1. - damping) * fy[i] * step

                        let dx = clamp -maxMove maxMove vx[i]
                        let dy = clamp -maxMove maxMove vy[i]
                        vx[i] <- dx
                        vy[i] <- dy
                        maxStepMove <- max maxStepMove (sqrt (square dx + square dy))
                        x[i] <- max leftMargin (x[i] + dx)
                        y[i] <- max topMargin (y[i] + dy)

                    if iteration % arrangeTiming.VisualizeEvery = 0 || iteration = arrangeTiming.Iterations - 1 then
                        if runSerial = arrangeRunSerial then
                            applyPositions()

                        do! Async.Sleep arrangeWeights.VisualizationDelayMs

                    if iteration > 3 && maxStepMove < arrangeTiming.StopWhenMaxMoveBelow then
                        isConverged <- true

                    iteration <- iteration + 1

                if runSerial = arrangeRunSerial then
                    applyPositions()

                    let requiredWidth =
                        nodes
                        |> Array.map (fun node -> node.X + node.Width)
                        |> Array.max
                        |> fun right -> right + leftMargin

                    let requiredHeight =
                        nodes
                        |> Array.map (fun node -> node.Y + node.Height)
                        |> Array.max
                        |> fun bottom -> bottom + topMargin

                    drawing.Width <- max drawing.Width requiredWidth
                    drawing.Height <- max drawing.Height requiredHeight

                    nodes |> Array.iter _.ClampToDrawing()
                    this.MarkGraphDirty()
        }
        |> Async.StartImmediate

    member this.DeleteSelectedCommand =
        SimpleCommand((fun _ -> this.DeleteSelectedElement()), (fun _ -> this.HasSelectedNodes))
        :> ICommand

    member this.ArrangeGraphCommand =
        SimpleCommand((fun _ -> this.ArrangeGraph()), (fun _ -> true))
        :> ICommand

    member _.ResetArrangeSettingsCommand =
        SimpleCommand((fun _ -> arrangeSettings.ResetDefaults()), (fun _ -> true))
        :> ICommand

    member this.RunCommand =
        SimpleCommand(
            (fun _ ->
                match this.ValidateGraph() with
                | Ok () ->
                    this.SetRunInProgress(true)
                    let cancellation = new CancellationTokenSource()
                    runCancellation <- Some cancellation
                    async {
                        try
                            let! fileDirectoryInputsResolved = this.ResolveFileDirectoryInputs()

                            if cancellation.IsCancellationRequested then
                                this.AppendGraphOutputLine("Run stopped")
                            elif fileDirectoryInputsResolved then
                                let generatedProgram = PipelineCodeGenerator.generateSavedGraph (this.ExportGraph())
                                this.AppendGeneratedProgram(generatedProgram)
                                do! this.BuildAndRunGeneratedProgram(generatedProgram) cancellation.Token
                            else
                                this.SetRunInProgress(false)
                        finally
                            runCancellation <- None
                            this.SetRunInProgress(false)
                            cancellation.Dispose()
                    }
                    |> Async.StartImmediate
                | Error message -> this.AppendGeneratedProgram(message)),
            (fun _ -> true))
        :> ICommand

    member this.StopRunCommand =
        SimpleCommand(
            (fun _ ->
                match runCancellation with
                | Some cancellation when not cancellation.IsCancellationRequested ->
                    this.AppendGraphOutputLine("Stopping")
                    cancellation.Cancel()
                    this.KillActiveRunProcess()
                | _ -> ()),
            (fun _ -> true))
        :> ICommand

    member _.ExportGraph() =
        let nodes = pipelineNodes () |> Seq.toArray

        let nodeIds =
            let ids = Dictionary<PipelineNodeViewModel, string>()

            nodes
            |> Array.iteri (fun index node -> ids.Add(node, $"node-{index + 1}"))

            ids

        let savedNodes =
            nodes
            |> Array.map (fun node ->
                { Id = nodeIds[node]
                  FunctionId = node.State.Definition.Id
                  X = node.X
                  Y = node.Y
                  Parameters = parameterValues node.State })

        let savedEdges =
            drawing.Connectors
            |> Seq.choose (fun connector ->
                match connector.Start, connector.End with
                | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                    match startPin.Parent, endPin.Parent with
                    | (:? PipelineNodeViewModel as startNode), (:? PipelineNodeViewModel as endNode) ->
                        match pinIndexByKind startPin.Kind startPin startNode, pinIndexByKind endPin.Kind endPin endNode with
                        | Some fromPort, Some toPort ->
                            Some
                                { FromNode = nodeIds[startNode]
                                  FromKind = PipelinePinKind.toString startPin.Kind
                                  FromPort = fromPort
                                  ToNode = nodeIds[endNode]
                                  ToKind = PipelinePinKind.toString endPin.Kind
                                  ToPort = toPort }
                        | _ -> None
                    | _ -> None
                | _ -> None)
            |> Seq.toArray

        let savedGraph =
            { Version = 1
              Nodes = savedNodes
              Edges = savedEdges }

        savedGraph

    member this.ExportGraphJson() =
        this.ExportGraph() |> PipelineGraphStorage.serialize

    member this.SaveGraph(path: string) =
        this.ExportGraph() |> PipelineGraphStorage.save path
        currentGraphPath <- Some path
        this.MarkGraphSaved()
        this.RaiseGraphStateChanged()

    member this.ImportGraph(savedGraph: SavedGraph) =
        drawing.Connectors.Clear()
        drawing.Nodes.Clear()
        this.SelectedNode <- null

        let loadedNodes =
            savedGraph.Nodes
            |> Array.map (fun savedNode ->
                let functionId = savedNode.FunctionId

                match BuiltInCatalog.tryFind functionId with
                | None -> invalidOp $"Unknown function id in saved graph: {savedNode.FunctionId}"
                | Some _ ->
                    let node =
                        PipelineNodeViewModel(
                            createState functionId,
                            (fun node -> this.SelectNodeFromEditor node),
                            moveSelectedNodesBy,
                            (fun () -> drawing.Width, drawing.Height),
                            (fun () -> this.MarkGraphDirty()),
                            removePinConnections,
                            refreshNodePins,
                            handleTypeChange)

                    watchState node.State
                    node.X <- savedNode.X
                    node.Y <- savedNode.Y
                    node.ClampToDrawing()
                    node.SyncMoveOrigin()
                    setParameterValues node.State savedNode.Parameters
                    SerialTransformNode.updateMethodParameterStates node.State
                    SourceImageNode.updateParameterVisibility node.State
                    node.RebuildPins()
                    drawing.Nodes.Add(node :> INode)
                    savedNode.Id, node)
            |> Map.ofArray

        for edge in savedGraph.Edges do
            match loadedNodes |> Map.tryFind edge.FromNode, loadedNodes |> Map.tryFind edge.ToNode with
            | Some fromNode, Some toNode
                when toNode.State.Definition.Id = "Expand"
                     && edge.ToKind <> "parameterInput"
                     && edge.ToPort = 0 ->
                let fromKind = PipelinePinKind.ofString edge.FromKind

                pinByKindIndex fromKind edge.FromPort fromNode
                |> Option.iter (fun outputPin ->
                    toNode.State.RecordType <- Some outputPin.Port.Type
                    toNode.RebuildPins())
            | _ -> ()

        for edge in savedGraph.Edges do
            match loadedNodes |> Map.tryFind edge.FromNode, loadedNodes |> Map.tryFind edge.ToNode with
            | Some fromNode, Some toNode ->
                let fromKind = PipelinePinKind.ofString edge.FromKind
                let toKind = PipelinePinKind.ofString edge.ToKind

                match pinByKindIndex fromKind edge.FromPort fromNode, pinByKindIndex toKind edge.ToPort toNode with
                | Some startPin, Some endPin when canConnectPins startPin endPin ->
                    addConnector startPin endPin
                    if toNode.State.Definition.Id = "Expand" && edge.ToKind <> "parameterInput" && edge.ToPort = 0 then
                        toNode.State.RecordType <- Some startPin.Port.Type
                        toNode.RebuildPins()
                        refreshNodePins toNode
                | Some _, Some _ ->
                    invalidOp $"Saved edge has incompatible port types: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{edge.ToPort}]"
                | _ ->
                    invalidOp $"Saved edge refers to a missing port: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{edge.ToPort}]"
            | _ ->
                invalidOp $"Saved edge refers to a missing node: {edge.FromNode} -> {edge.ToNode}"

        this.MarkGraphSaved()
        this.RaiseGraphStateChanged()

    member this.ImportGraphJson(json: string) =
        json |> PipelineGraphStorage.deserialize |> this.ImportGraph

    member this.LoadGraph(path: string) =
        path |> PipelineGraphStorage.load |> this.ImportGraph
        currentGraphPath <- Some path
        this.RaiseGraphStateChanged()

    member _.HasGraph =
        drawing.Nodes.Count > 0 || drawing.Connectors.Count > 0

    member _.IsGraphDirty = graphDirty

    member _.CurrentGraphPath = currentGraphPath

    member _.HasGraphFile = currentGraphPath.IsSome

    member this.CanSaveGraph = this.HasGraph && this.HasGraphFile && this.IsGraphDirty

    member this.CanSaveGraphAs = this.HasGraph

    member this.CanClearGraph = this.HasGraph

    member this.CanRunGraph = this.HasGraph && not isRunInProgress

    member _.CanStopRun = isRunInProgress

    member _.SuggestedGraphFileName =
        currentGraphPath
        |> Option.map Path.GetFileName
        |> Option.defaultValue "pipeline.json"

    member _.CurrentGraphFileName =
        currentGraphPath
        |> Option.map Path.GetFileName
        |> Option.defaultValue ""

    member this.SetCurrentGraphPath(path: string) =
        currentGraphPath <- Some path
        this.RaiseGraphStateChanged()

    member this.ClearCurrentGraphPath() =
        currentGraphPath <- None
        this.RaiseGraphStateChanged()

    member this.RaiseGraphStateChanged() =
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasGraph))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CurrentGraphPath))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasGraphFile))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanSaveGraph))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanSaveGraphAs))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanClearGraph))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanRunGraph))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SuggestedGraphFileName))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CurrentGraphFileName))

    member this.MarkGraphDirty() =
        if not graphDirty then
            graphDirty <- true
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.IsGraphDirty))
            this.RaiseGraphStateChanged()

    member this.MarkGraphSaved() =
        if graphDirty then
            graphDirty <- false
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.IsGraphDirty))
            this.RaiseGraphStateChanged()

    member this.ClearGraph() =
        let shouldMarkDirty = this.HasGraph || graphDirty
        drawing.Connectors.Clear()
        drawing.Nodes.Clear()
        this.SelectedNode <- null
        graphOutput <- ""
        currentGraphPath <- None
        this.RaiseGraphOutputChanged()
        this.RaiseGraphStateChanged()

        if shouldMarkDirty then
            this.MarkGraphDirty()

    member this.AddElement(functionId: string) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- min (max 0. (drawing.Width - node.Width)) (24. + float (drawing.Nodes.Count % 6) * 118.)
        node.Y <- min (max 0. (drawing.Height - node.Height)) (24. + float (drawing.Nodes.Count / 6) * 72.)
        node.SyncMoveOrigin()

        drawing.Nodes.Add(node :> INode)
        node.RebuildPins()
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddElementAt(functionId: string, x: float, y: float) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.
        node.ClampToDrawing()
        node.SyncMoveOrigin()

        drawing.Nodes.Add(node :> INode)
        node.RebuildPins()
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddPaletteDragElementAt(functionId: string, x: float, y: float, isOutsideGraph: bool) =
        let node = createNode drawing.Nodes.Count functionId
        node.State.IsPaletteDragOutside <- isOutsideGraph
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.
        node.SyncMoveOrigin()

        drawing.Nodes.Add(node :> INode)
        node.RebuildPins()
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.MoveSelectedElementTo(x: float, y: float, shouldClamp: bool, isPaletteDragOutside: bool) =
        if not (isNull selectedNode) then
            selectedNode.State.IsPaletteDragOutside <- isPaletteDragOutside
            selectedNode.X <- x - selectedNode.Width / 2.
            selectedNode.Y <- y - selectedNode.Height / 2.

            if shouldClamp then
                selectedNode.ClampToDrawing()

        selectedNode.SyncMoveOrigin()
        this.MarkGraphDirty()

    member this.DuplicateSelectedElements() =
        let selected =
            if selectedNodes.Count > 0 then
                selectedNodes |> Seq.toArray
            elif not (isNull selectedNode) then
                [| selectedNode |]
            else
                Array.empty

        if selected.Length > 0 then
            let offset = 32.
            let clones = Dictionary<PipelineNodeViewModel, PipelineNodeViewModel>(HashIdentity.Reference)

            for original in selected do
                let clone = createNode drawing.Nodes.Count original.State.Definition.Id
                clone.X <- original.X + offset
                clone.Y <- original.Y + offset
                clone.State.RecordType <- original.State.RecordType
                setParameterValues clone.State (parameterValues original.State)
                SerialTransformNode.updateMethodParameterStates clone.State
                SourceImageNode.updateParameterVisibility clone.State
                clone.ClampToDrawing()
                clone.SyncMoveOrigin()
                drawing.Nodes.Add(clone :> INode)
                clone.RebuildPins()
                clones.Add(original, clone)

            let selectedSet = HashSet<PipelineNodeViewModel>(selected, HashIdentity.Reference)

            let internalConnectors =
                drawing.Connectors
                |> Seq.choose (fun connector ->
                    match connector.Start, connector.End with
                    | (:? PipelinePinViewModel as startPin), (:? PipelinePinViewModel as endPin) ->
                        match startPin.Parent, endPin.Parent with
                        | (:? PipelineNodeViewModel as startNode), (:? PipelineNodeViewModel as endNode)
                            when selectedSet.Contains startNode && selectedSet.Contains endNode ->
                            match pinIndexByKind startPin.Kind startPin startNode, pinIndexByKind endPin.Kind endPin endNode with
                            | Some fromPort, Some toPort ->
                                Some(startNode, startPin.Kind, fromPort, endNode, endPin.Kind, toPort)
                            | _ ->
                                None
                        | _ ->
                            None
                    | _ ->
                        None)
                |> Seq.toArray

            for startNode, startKind, fromPort, endNode, endKind, toPort in internalConnectors do
                match clones.TryGetValue startNode, clones.TryGetValue endNode with
                | (true, clonedStartNode), (true, clonedEndNode) ->
                    match pinByKindIndex startKind fromPort clonedStartNode, pinByKindIndex endKind toPort clonedEndNode with
                    | Some startPin, Some endPin when canConnectPins startPin endPin ->
                        addConnector startPin endPin
                    | _ ->
                        ()
                | _ ->
                    ()

            this.SelectNodes(clones.Values)
            this.MarkGraphDirty()

    member this.DeleteSelectedElement() =
        let selected =
            if selectedNodes.Count > 0 then
                selectedNodes |> Seq.toArray
            elif not (isNull selectedNode) then
                [| selectedNode |]
            else
                Array.empty

        if selected.Length > 0 then
            let nodes = pipelineNodes () |> Seq.toArray
            let currentIndex =
                nodes
                |> Array.tryFindIndex (fun node -> selected |> Array.exists (fun selectedNode -> Object.ReferenceEquals(node, selectedNode)))
                |> Option.defaultValue 0

            let pinsToRemove =
                selected
                |> Array.collect (fun node -> node.Pins |> Seq.toArray)

            let connectorsToRemove =
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    pinsToRemove
                    |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
                |> Seq.toArray

            for connector in connectorsToRemove do
                drawing.Connectors.Remove(connector) |> ignore

            for node in selected do
                drawing.Nodes.Remove(node) |> ignore

            let remaining = pipelineNodes () |> Seq.toArray
            if remaining.Length > 0 then
                this.SelectedNode <- remaining[min currentIndex (remaining.Length - 1)]
            else
                this.SelectedNode <- null

            this.MarkGraphDirty()

    member _.ValidateGraph() =
        let shouldRequirePin (pin: IPin) =
            match pin with
            | :? PipelinePinViewModel as pipelinePin ->
                match pipelinePin.Kind with
                | ParameterInput -> pipelinePin.IsActive
                | ScalarOutput -> false
                | ReducerOutput -> false
                | DataInput
                | DataOutput -> true
            | _ -> true

        let missingPins =
            drawing.Nodes
            |> Seq.collect (fun node ->
                node.Pins
                |> Seq.filter (fun pin -> shouldRequirePin pin && not (drawing.IsPinConnected(pin)))
                |> Seq.map (fun pin -> $"{node.Name}.{pin.Name}"))
            |> Seq.toArray

        if missingPins.Length = 0 then
            Ok ()
        else
            let message =
                missingPins
                |> Seq.map (fun pin -> $"// - {pin}")
                |> String.concat Environment.NewLine

            Error($"// Cannot generate F# yet. Connect every input and output pin first.{Environment.NewLine}{message}")

    member _.SetDrawingSize(width: float, height: float) =
        if width > 0. && height > 0. then
            drawing.Width <- width
            drawing.Height <- height

            pipelineNodes ()
            |> Seq.iter _.ClampToDrawing()

    member this.DeleteSelectedElementIfInTrashZone(trashWidth: float, trashHeight: float, margin: float) =
        let selected = this.SelectedNodes

        if selected.Length > 0 then
            let trashLeft = max 0. (drawing.Width - trashWidth - margin)
            let trashTop = max 0. (drawing.Height - trashHeight - margin)

            let selectionRight =
                selected
                |> Array.map (fun node -> node.X + node.Width)
                |> Array.max

            let selectionBottom =
                selected
                |> Array.map (fun node -> node.Y + node.Height)
                |> Array.max

            if selectionRight >= trashLeft && selectionBottom >= trashTop then
                this.DeleteSelectedElement()

    interface IGraphWindowController with
        member this.SetDrawingSize width height =
            this.SetDrawingSize(width, height)

        member this.MoveSelectionBy dx dy =
            this.MoveSelectionBy(dx, dy)

        member this.DeleteSelectedElementIfInTrashZone trashWidth trashHeight margin =
            this.DeleteSelectedElementIfInTrashZone(trashWidth, trashHeight, margin)

    member private this.RaiseGraphOutputChanged() =
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.GraphOutput))
        this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.GeneratedProgram))
