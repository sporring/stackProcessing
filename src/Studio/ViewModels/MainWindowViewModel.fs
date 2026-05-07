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
        | "parameterInput" -> ParameterInput
        | "scalarOutput" -> ScalarOutput
        | "reducerOutput" -> ReducerOutput
        | "dataOutput" -> DataOutput
        | _ -> DataInput

    let isInput kind =
        match kind with
        | DataInput
        | ParameterInput -> true
        | DataOutput
        | ScalarOutput
        | ReducerOutput -> false

    let isOutput kind = not (isInput kind)

type PipelinePinViewModel(alignment: PinAlignment, port: Port, kind: PipelinePinKind, ?parameterKey: string) =
    inherit PinViewModel()

    let mutable isActive = kind <> ParameterInput

    member _.Port = port
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

    member _.TrianglePoints =
        match kind with
        | ParameterInput
        | ScalarOutput
        | ReducerOutput -> "0,0 14,0 7,14"
        | DataInput -> "14,0 0,7 14,14"
        | DataOutput -> "0,0 14,7 0,14"

    member _.IsInput = PipelinePinKind.isInput kind
    member _.IsOutput = PipelinePinKind.isOutput kind

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
        functionId = "Read" || functionId = "ReadRandom" || functionId = "ReadRange" || functionId = "ReadSlab" || functionId = "ReadZarrSlab" || functionId = "ReadNexusSlab"

    let hasOutputTitle functionId =
        functionId = "Write" || functionId = "WriteThrough" || functionId = "WriteInSlabs" || functionId = "WriteZarr" || functionId = "WriteNexus"

    let hasFormatParameter functionId =
        (hasInputTitle functionId && functionId <> "ReadZarrSlab" && functionId <> "ReadNexusSlab")
        || (hasOutputTitle functionId && functionId <> "WriteZarr" && functionId <> "WriteNexus")
        || functionId = "WriteSlabSlices"
        || functionId = "GetStackInfo"
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

    let suffixOptionsFor functionId =
        if hasInputTitle functionId || functionId = "GetStackInfo" || functionId = "GetChunkInfo" || functionId = "ResampleAffineTrilinearSlices" then
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
        | "ReadZarrSlab" -> [ UInt8; UInt16 ]
        | "ReadNexusSlab" -> [ UInt8; Int8; UInt16; Int16; UInt32; Int32; Float32; Float64 ]
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

    let title (state: PipelineNodeState) =
        let parameterText =
            state.Parameters
            |> Seq.exists (fun parameter -> parameter.Key = "input")
            |> function
                | true -> parameterTitle "input" "input" state
                | false -> parameterTitle "output" "output" state

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

module private ScalarImageOperationNode =
    let typeOptions = SourceImageNode.typeOptions

    let operationOptions = [ "+"; "-"; "*"; "/" ]

    let isOperation functionId =
        functionId = "ImageOpScalar"
        || functionId = "ScalarOpImage"
        || functionId = "AddNormalNoise"

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
    let typedImageFunctionIds =
        [ "Clamp"
          "ShiftScale"
          "IntensityStretch"
          "CreatePadding"
          "Crop"
          "Median"
          "Bilateral"
          "GradientMagnitude"
          "SobelEdge"
          "Laplacian"
          "ImageComparison"
          "Mask"
          "GrayscaleErode"
          "GrayscaleDilate"
          "GrayscaleOpening"
          "GrayscaleClosing"
          "WhiteTopHat"
          "BlackTopHat"
          "MorphologicalGradient"
          "OtsuThreshold"
          "MomentsThreshold"
          "LabelContour"
          "ChangeLabel" ]
        |> Set.ofList

    let typeOptions = SourceImageNode.typeOptions
    let comparisonOptions = [ ">"; ">="; "<"; "<="; "="; "<>"; "!=" ]
    let maskLogicOptions = [ "and"; "or"; "xor" ]
    let boolOptions = [ "false"; "true" ]

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

module private ChartNode =
    let kindOptions =
        [ "Scatter"; "Line"; "Bar"; "Column"; "Area"; "Pie"; "Doughnut" ]

    let title (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun parameter -> parameter.Key = "kind")
        |> Option.map _.Value
        |> Option.filter (fun value -> kindOptions |> List.contains value)
        |> Option.defaultValue "Column"

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
    refreshNodePins: PipelineNodeViewModel -> unit) as this =
    inherit NodeViewModel()

    let mutable lastX = 0.
    let mutable lastY = 0.
    let mutable suppressGroupMove = false
    let pinSize = 14.
    let pinHalfSize = pinSize / 2.

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
        pin.X <- x
        pin.Y <- y
        pin.Width <- pinSize
        pin.Height <- pinSize
        pin.Alignment <- alignment
        this.Pins.Add(pin :> IPin)
        pin :> IPin

    let nodeHeight =
        let portCount =
            if state.Definition.Id = "ComputeStats" || state.Definition.Id = "ComponentTranslationTable" then
                state.Definition.Inputs.Length
            else
                max state.Definition.Inputs.Length state.Definition.Outputs.Length

        max 48. (20. + 22. * float (max 1 portCount))

    let verticalPinPosition index count =
        if count <= 1 then
            nodeHeight / 2.
        else
            let spacing = 22.
            let totalHeight = spacing * float (count - 1)
            (nodeHeight - totalHeight) / 2. + spacing * float index

    do
        this.Name <- state.Title
        this.Width <-
            match state.Definition.Id with
            | "ComputeStats"
            | "GetStackInfo"
            | "GetChunkInfo"
            | "Quantiles" -> 170.
            | "ComponentTranslationTable"
            | "CollapseComponentLabels" -> 180.
            | "ResampleAffineTrilinearSlices" -> 220.
            | _ -> 110.
        this.Height <- nodeHeight
        this.Content <- PipelineNodeContent(state.Title, state, this.Width, this.Height, fun () -> selectNode this)
        this.Pins <- ObservableCollection<IPin>()

        state.Parameters
        |> Seq.iter (fun parameter ->
            parameter.PropertyChanged.Add(fun args ->
                if args.PropertyName = nameof parameter.UseInput then
                    this.SyncParameterPinVisibility()
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
                elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "operation" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarImageOperationNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "value" && args.PropertyName = nameof parameter.Value then
                    state.Title <- ScalarImageOperationNode.title state
                    this.Name <- state.Title
                    markGraphDirty()
                elif (state.Definition.Id = "Scalar" || state.Definition.Id = "ScalarOp" || state.Definition.Id = "Read" || state.Definition.Id = "ReadRandom" || state.Definition.Id = "ReadRange" || state.Definition.Id = "ReadSlab" || state.Definition.Id = "ReadZarrSlab" || state.Definition.Id = "ReadNexusSlab" || state.Definition.Id = "Zero" || state.Definition.Id = "CreateByEuler2DTransform" || state.Definition.Id = "Threshold" || state.Definition.Id = "ImageOpImage" || state.Definition.Id = "Resize" || state.Definition.Id = "Resample" || state.Definition.Id = "ResampleAffineTrilinearSlices" || HighValueFilterNode.typedImageFunctionIds.Contains state.Definition.Id || ScalarImageOperationNode.isOperation state.Definition.Id) && parameter.Key = "type" && args.PropertyName = nameof parameter.Value then
                    if state.Definition.Id = "Scalar" then
                        ScalarNode.ensureValueMatchesType state
                        state.Title <- ScalarNode.title state
                        this.Name <- state.Title

                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()
                elif SourceImageNode.hasFormatParameter state.Definition.Id && parameter.Key = "suffix" && args.PropertyName = nameof parameter.Value then
                    if SourceImageNode.hasInputTitle state.Definition.Id
                       || state.Definition.Id = "Zero"
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
                elif state.Definition.Id = "Cast" && (parameter.Key = "sourceType" || parameter.Key = "targetType") && args.PropertyName = nameof parameter.Value then
                    this.RebuildPins()
                    refreshNodePins this
                    markGraphDirty()))

        this.InitializePins()

    member private this.RemoveConnectionsForPin(pin: IPin) =
        removePinConnections [ pin ]

    member private this.TryFindParameterPin(parameterKey: string) =
        this.Pins
        |> Seq.tryPick (function
            | :? PipelinePinViewModel as pin when pin.Kind = ParameterInput && pin.ParameterKey = parameterKey ->
                Some(pin :> IPin)
            | _ -> None)

    member private _.ParameterPinIsVisible(parameter: PipelineParameterViewModel) =
        parameter.UseInput
        && (state.Definition.Id <> "Print" || PrintNode.inputIsVisible state parameter.Key)
        || (state.Definition.Id = "CollapseComponentLabels" && parameter.Key = "translationTable")

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
            pin.Width <- 0.
            pin.Height <- 0.

            match pin with
            | :? PipelinePinViewModel as parameterPin -> parameterPin.SetActive(false)
            | _ -> ()

    member private this.ParameterPinX(index: int, count: int) =
        let n = float (index + 1)
        n * this.Width / float (count + 1)

    member private this.AddParameterPin(index: int, count: int, parameter: PipelineParameterViewModel) =
        let x = this.ParameterPinX(index, count)
        let port =
            if state.Definition.Id = "ScalarOp" && (parameter.Key = "a" || parameter.Key = "b") then
                ScalarOpNode.scalarPort parameter.Key state
            elif state.Definition.Id = "ScalarFunction" && parameter.Key = "a" then
                ScalarFunctionNode.scalarPort
            elif ScalarImageOperationNode.isOperation state.Definition.Id && parameter.Key = "value" then
                ScalarImageOperationNode.valuePort state
            elif state.Definition.Id = "CollapseComponentLabels" && parameter.Key = "translationTable" then
                PortMapping.customParameterPort parameter.Key "TranslationTable"
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
                match visibleParameterIndexes |> Set.toList |> List.tryFindIndex ((=) index) with
                | Some visibleIndex ->
                    pin.X <- this.ParameterPinX(visibleIndex, visibleParameterIndexes.Count)
                    pin.Y <- 0.
                | None ->
                    ()

                this.SetParameterPinVisibility(parameter, pin)
            | None ->
                this.AddParameterPin(0, 1, parameter))

    member private this.InitializePins() =
        this.Pins.Clear()

        let inputs, outputs =
            match state.Definition.Id with
            | "Scalar" -> state.Definition.Inputs, [ ScalarNode.outputPort state ]
            | "FileDirectory" -> state.Definition.Inputs, state.Definition.Outputs
            | "ScalarOp" -> state.Definition.Inputs, [ ScalarOpNode.outputPort state ]
            | "ScalarFunction" -> state.Definition.Inputs, [ ScalarFunctionNode.outputPort ]
            | "Read"
            | "ReadRandom"
            | "ReadRange"
            | "ReadSlab"
            | "ReadZarrSlab"
            | "ReadNexusSlab"
            | "Zero"
            | "CreateByEuler2DTransform" -> state.Definition.Inputs, [ SourceImageNode.outputPort state ]
            | "Write"
            | "WriteThrough"
            | "WriteInSlabs"
            | "WriteZarr"
            | "WriteNexus" ->
                [ SourceImageNode.writeInputPort state ], state.Definition.Outputs
            | "ImageOpImage" -> PairOperationNode.ports state
            | "Cast" -> CastNode.ports state
            | functionId when ScalarImageOperationNode.isOperation functionId -> ScalarImageOperationNode.ports state
            | "Threshold" -> ThresholdNode.ports state
            | "Quantiles" -> state.Definition.Inputs, QuantilesNode.outputPorts state
            | _ -> state.Definition.Inputs, state.Definition.Outputs

        inputs
        |> List.iteri (fun portIndex port ->
            addPipelinePin 0. (verticalPinPosition portIndex inputs.Length) PinAlignment.Left DataInput None port |> ignore)

        outputs
        |> List.iteri (fun portIndex port ->
            let kind =
                match state.Definition.Id with
                | "Scalar"
                | "FileDirectory"
                | "ScalarOp"
                | "ScalarFunction" -> ScalarOutput
                | "ComputeStats"
                | "GetStackInfo"
                | "GetChunkInfo"
                | "ComponentTranslationTable"
                | "HistogramData"
                | "Quantiles" -> ReducerOutput
                | _ -> DataOutput

            let alignment =
                if kind = ScalarOutput || kind = ReducerOutput then PinAlignment.Bottom else PinAlignment.Right

            let x =
                if kind = ReducerOutput then
                    let spacing = this.Width / float (outputs.Length + 1)
                    spacing * float (portIndex + 1)
                elif kind = ScalarOutput then
                    this.Width / 2.
                else
                    this.Width

            let y =
                if kind = ScalarOutput || kind = ReducerOutput then
                    nodeHeight
                else
                    verticalPinPosition portIndex outputs.Length

            addPipelinePin x y alignment kind None port |> ignore)

        if state.Definition.Id = "Tap" then
            addPipelinePin
                (this.Width / 2.)
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
        this.InitializePins()

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
    let mutable selectedNode: PipelineNodeViewModel = null
    let selectedNodes = HashSet<PipelineNodeViewModel>(HashIdentity.Reference)
    let mutable graphOutput = ""
    let mutable isRunInProgress = false
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
                | "ReadZarrSlab", "type" ->
                    let supported = [ "UInt8"; "UInt16" ] |> Set.ofList
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, supported |> Set.contains value))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "ReadNexusSlab", "type" ->
                    let supported = [ "UInt8"; "Int8"; "UInt16"; "Int16"; "UInt32"; "Int32"; "Float32"; "Float64" ] |> Set.ofList
                    let options =
                        SourceImageNode.typeOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, supported |> Set.contains value))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("Read" | "ReadRandom" | "ReadRange" | "ReadSlab" | "Zero" | "CreateByEuler2DTransform" | "ResampleAffineTrilinearSlices"), "type" ->
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
                | ("DiscreteGaussian" | "Convolve"), "outputRegionMode" ->
                    let options =
                        [ "None"; "Valid"; "Same" ]
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | ("DiscreteGaussian" | "Convolve"), "boundaryCondition" ->
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
                | ("BinaryContour" | "BinaryOpeningByReconstruction" | "BinaryClosingByReconstruction" | "BinaryReconstructionByDilation" | "BinaryReconstructionByErosion" | "LabelContour"), "fullyConnected" ->
                    let options =
                        HighValueFilterNode.boolOptions
                        |> List.map (fun value -> ParameterOptionViewModel(value, value, true))

                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, options, false)
                | "Quantiles", key when key.StartsWith("useQ", StringComparison.Ordinal) ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, canUseInput = false)
                | "Quantiles", "histogram" ->
                    PipelineParameterViewModel(parameter.Label, parameter.Key, parameter.DefaultValue, parameter.Type, forceUseInput = true)
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
                | "Chart", "kind" ->
                    let options =
                        ChartNode.kindOptions
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

    let hasConnectionRequiringFixedDataOutput (node: PipelineNodeViewModel) =
        node.Pins
        |> Seq.exists (function
            | :? PipelinePinViewModel as pin when pin.Kind = DataOutput ->
                drawing.Connectors
                |> Seq.exists (fun connector ->
                    Object.ReferenceEquals(connector.Start, pin)
                    && not (isNumberImagePin connector.End))
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
            || node.State.Definition.Id = "ReadZarrSlab"
            || node.State.Definition.Id = "ReadNexusSlab"
            || node.State.Definition.Id = "Zero"
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
                    option.IsEnabled <- supported && (not isConnected || option.Value = parameter.Value)

                if not (supportedTypes |> Set.contains parameter.Value) && not isConnected then
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

    let refreshImageFormatOptions () =
        pipelineNodes ()
        |> Seq.filter (fun node -> SourceImageNode.hasFormatParameter node.State.Definition.Id)
        |> Seq.iter (fun node ->
            let requiredType =
                if SourceImageNode.hasOutputTitle node.State.Definition.Id then
                    connectedWriteInputType node
                elif SourceImageNode.hasInputTitle node.State.Definition.Id then
                    Some(SourceImageNode.selectedType node.State)
                else
                    None

            node.State.Parameters
            |> Seq.tryFind (fun parameter -> parameter.Key = "suffix")
            |> Option.iter (fun parameter ->
                for option in parameter.Options do
                    option.IsEnabled <-
                        requiredType
                        |> Option.forall (ImageFileFormat.supports option.Value)

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
            elif outputPin.Kind = DataOutput && inputPin.Kind = DataInput then
                let baseCompatible = PortType.canConnect outputPin.Port.Type inputPin.Port.Type
                let formatCompatible =
                    match inputPin.Parent, outputPin.Port.Type with
                    | :? PipelineNodeViewModel as inputNode, Image Number
                        when SourceImageNode.hasOutputTitle inputNode.State.Definition.Id ->
                        true
                    | :? PipelineNodeViewModel as inputNode, Image numericType
                        when inputNode.State.Definition.Id = "WriteZarr" ->
                        numericType = UInt8 || numericType = UInt16
                    | :? PipelineNodeViewModel as inputNode, Image numericType
                        when SourceImageNode.hasOutputTitle inputNode.State.Definition.Id ->
                        SourceImageNode.selectedSuffix inputNode.State
                        |> fun suffix -> ImageFileFormat.supports suffix numericType
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
                    Some 0

            let nodeEndpoint (pin: IPin) =
                match pin with
                | :? PipelinePinViewModel as pipelinePin when Object.ReferenceEquals(pipelinePin.Parent, node) ->
                    dynamicPinIndex pipelinePin
                    |> Option.map (fun index -> pipelinePin.Kind, index)
                | _ -> None

            let connectors =
                drawing.Connectors
                |> Seq.filter (fun connector -> Object.ReferenceEquals(connector.Start.Parent, node) || Object.ReferenceEquals(connector.End.Parent, node))
                |> Seq.toArray

            let connectorSnapshots =
                connectors
                |> Array.map (fun connector -> connector, nodeEndpoint connector.Start, nodeEndpoint connector.End)

            for connector in connectors do
                drawing.Connectors.Remove(connector) |> ignore

            let index =
                drawing.Nodes
                |> Seq.tryFindIndex (fun candidate -> Object.ReferenceEquals(candidate, node))
                |> Option.defaultValue (drawing.Nodes.Count - 1)

            drawing.Nodes.Remove(node :> INode) |> ignore
            drawing.Nodes.Insert(index, node :> INode)

            Dispatcher.UIThread.Post(
                (fun () ->
                    for connector, startEndpoint, endEndpoint in connectorSnapshots do
                        startEndpoint
                        |> Option.bind (fun (kind, index) -> pinByKindIndex kind index node)
                        |> Option.iter (fun pin -> connector.Start <- pin)

                        endEndpoint
                        |> Option.bind (fun (kind, index) -> pinByKindIndex kind index node)
                        |> Option.iter (fun pin -> connector.End <- pin)

                        if canConnectPins connector.Start connector.End then
                            drawing.Connectors.Add(connector) |> ignore),
                DispatcherPriority.Background)

    let createNode index functionId =
        let node =
            PipelineNodeViewModel(
                createState functionId,
                (fun node -> this.SelectNodeFromEditor node),
                moveSelectedNodesBy,
                (fun () -> drawing.Width, drawing.Height),
                (fun () -> this.MarkGraphDirty()),
                removePinConnections,
                refreshNodePins)

        watchState node.State

        node.X <- float (24 + index * 118)
        node.Y <- 66.
        node.ClampToDrawing()
        node.SyncMoveOrigin()
        node

    let addConnector startPin endPin =
        let connector = ConnectorViewModel()
        connector.Start <- startPin
        connector.End <- endPin
        connector.Orientation <- connectorOrientation startPin endPin
        drawing.Connectors.Add(connector :> IConnector)
        this.MarkGraphDirty()

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
                refreshScalarTypeOptions()
                refreshScalarOpTypeOptions()
                refreshImageFormatOptions()
                refreshSourceImageTypeOptions()
                refreshCastTypeOptions()
                refreshPairOperationTypeOptions()
                refreshScalarImageOperationTypeOptions()
                refreshThresholdTypeOptions()
                this.RaiseGraphStateChanged()
                this.MarkGraphDirty())
        | _ -> ()

        //addSeedNodes()

    member _.Editor = editor
    member _.PaletteGroups = paletteGroups

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

    member this.SelectSingleNode(node: PipelineNodeViewModel) =
        this.SelectedNode <- node

    member this.SelectNodeFromEditor(node: PipelineNodeViewModel) =
        if not (isNull node) && node.State.IsSelected && selectedNodes.Count > 1 then
            selectedNode <- node
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
            this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))
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
            else
                addSelectedNode node
                selectedNode <- node
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedNode))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.SelectedElement))
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.HasSelectedElement))

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

    member this.MoveSelectionBy(dx: float, dy: float) =
        let selected = selectedNodes |> Seq.toArray

        if selected.Length > 0 && (dx <> 0. || dy <> 0.) then
            let clampedDx, clampedDy = clampSelectionDelta selected dx dy
            applySelectionDelta selected clampedDx clampedDy

    member _.GraphOutput = graphOutput

    member _.GeneratedProgram = graphOutput

    member private this.SetGraphOutput(text: string) =
        graphOutput <- text
        this.RaiseGraphOutputChanged()

    member this.AppendGraphOutput(text: string) =
        let separator =
            if String.IsNullOrEmpty graphOutput || text.StartsWith(Environment.NewLine, StringComparison.Ordinal) then
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

    member private this.AppendGraphOutputLineOnUi(text: string) =
        Dispatcher.UIThread.Post(fun () -> this.AppendGraphOutputLine(text))

    member private this.SetRunInProgress(value: bool) =
        if Dispatcher.UIThread.CheckAccess() then
            if isRunInProgress <> value then
                isRunInProgress <- value
                this.OnPropertyChanged(PropertyChangedEventArgs(nameof this.CanRunGraph))
        else
            Dispatcher.UIThread.Post(fun () -> this.SetRunInProgress(value))

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

    member private this.RunProcess(phase: string) (fileName: string) (arguments: string list) (workingDirectory: string) =
        async {
            this.AppendGraphOutputLineOnUi(phase)

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
                    let mutable keepReading = true

                    while keepReading do
                        let! line = reader.ReadLineAsync() |> Async.AwaitTask

                        if isNull line then
                            keepReading <- false
                        else
                            this.AppendGraphOutputLineOnUi(line)
                }

            if not (proc.Start()) then
                return -1
            else
                let! output = readLines proc.StandardOutput |> Async.StartChild
                let! error = readLines proc.StandardError |> Async.StartChild

                do! proc.WaitForExitAsync() |> Async.AwaitTask
                do! output
                do! error

                return proc.ExitCode
        }

    member private this.BuildAndRunGeneratedProgram(generatedProgram: string) =
        async {
            try
                this.SetRunInProgress(true)
                this.AppendGraphOutputLineOnUi("Compiling")

                let projectPath = this.EnsureRunProject(generatedProgram.Contains("open Plotly.NET", StringComparison.Ordinal))
                this.WriteRunProgram(generatedProgram)

                let dotnet = this.DotnetExecutable()

                let! buildExitCode =
                    this.RunProcess
                        "Building"
                        dotnet
                        [ "build"; projectPath; "--configuration"; "Release"; "--nologo"; "--verbosity"; "minimal" ]
                        runProjectDirectory

                if buildExitCode = 0 then
                    let runWorkingDirectory = this.GraphRunWorkingDirectory()
                    this.AppendGraphOutputLineOnUi($"Working directory: {runWorkingDirectory}")

                    let! runExitCode =
                        this.RunProcess
                            "Run"
                            dotnet
                            [ "run"; "--configuration"; "Release"; "--no-build"; "--project"; projectPath ]
                            runWorkingDirectory

                    if runExitCode = 0 then
                        this.AppendGraphOutputLineOnUi("Completed")
                    else
                        this.AppendGraphOutputLineOnUi($"Run failed with exit code {runExitCode}")
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
        SimpleCommand((fun _ -> this.AddElement("DiscreteGaussian")), (fun _ -> true)) :> ICommand

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

            let hasDataPin (node: PipelineNodeViewModel) =
                node.Pins
                |> Seq.exists (function
                    | :? PipelinePinViewModel as pin -> pin.Kind = DataInput || pin.Kind = DataOutput
                    | _ -> false)

            let columnSpacing = 190.
            let rowSpacing = 108.
            let leftMargin = 32.
            let topMargin = 32.

            let isParameterEdge (startPin: PipelinePinViewModel) (endPin: PipelinePinViewModel) =
                (startPin.Kind = ScalarOutput || startPin.Kind = ReducerOutput) && endPin.Kind = ParameterInput

            let isDataEdge (startPin: PipelinePinViewModel) (endPin: PipelinePinViewModel) =
                startPin.Kind = DataOutput && endPin.Kind = DataInput

            let xRanks = Dictionary<PipelineNodeViewModel, int>()
            let yRanks = Dictionary<PipelineNodeViewModel, int>()

            nodes
            |> Array.iter (fun node ->
                xRanks[node] <- 0
                yRanks[node] <- 0)

            let updateRank (ranks: Dictionary<PipelineNodeViewModel, int>) startNode endNode delta =
                let candidate = ranks[startNode] + delta

                if candidate > ranks[endNode] then
                    ranks[endNode] <- candidate

            for _ in 1 .. nodes.Length do
                for startPin, endPin, startNode, endNode in connectorEdges do
                    if isDataEdge startPin endPin || isParameterEdge startPin endPin then
                        updateRank xRanks startNode endNode 1
                    if isParameterEdge startPin endPin then
                        updateRank yRanks startNode endNode 1

            let incomingCount node =
                connectorEdges
                |> Array.sumBy (fun (_, _, _, endNode) ->
                    if Object.ReferenceEquals(endNode, node) then 1 else 0)

            let outgoingCount node =
                connectorEdges
                |> Array.sumBy (fun (_, _, startNode, _) ->
                    if Object.ReferenceEquals(startNode, node) then 1 else 0)

            let predecessors node =
                connectorEdges
                |> Array.choose (fun (startPin, endPin, startNode, endNode) ->
                    if (isDataEdge startPin endPin || isParameterEdge startPin endPin)
                       && Object.ReferenceEquals(endNode, node) then
                        Some startNode
                    else
                        None)

            let incomingEdges node =
                connectorEdges
                |> Array.filter (fun (_, _, _, endNode) -> Object.ReferenceEquals(endNode, node))

            let columns =
                nodes
                |> Array.groupBy (fun node -> xRanks[node])
                |> Array.sortBy fst

            let rowHints = Dictionary<PipelineNodeViewModel, float>()

            for depth, columnNodes in columns do
                let ordered =
                    columnNodes
                    |> Array.sortBy (fun node ->
                        let predecessorRows =
                            predecessors node
                            |> Array.choose (fun predecessor ->
                                match rowHints.TryGetValue predecessor with
                                | true, row -> Some row
                                | _ -> None)

                        let predecessorRow =
                            if predecessorRows.Length = 0 then
                                node.Y / rowSpacing
                            else
                                predecessorRows |> Array.average

                        predecessorRow,
                        yRanks[node],
                        incomingEdges node |> Array.exists (fun (startPin, endPin, _, _) -> isParameterEdge startPin endPin),
                        incomingCount node,
                        -outgoingCount node,
                        node.Y,
                        node.X,
                        node.State.Title)

                ordered
                |> Array.iteri (fun index node ->
                    node.X <- leftMargin + float depth * columnSpacing
                    node.Y <- topMargin + float index * rowSpacing
                    rowHints[node] <- float index)

            let settleColumnCollisions () =
                columns
                |> Array.iter (fun (_, columnNodes) ->
                    let mutable previousBottom = Double.NegativeInfinity

                    columnNodes
                    |> Array.sortBy (fun node -> node.Y, node.X, node.State.Title)
                    |> Array.iter (fun node ->
                        let minimumY = previousBottom + 32.

                        if node.Y < minimumY then
                            node.Y <- minimumY

                        previousBottom <- node.Y + node.Height))

            for _ in 1 .. max 1 nodes.Length do
                for startPin, endPin, startNode, endNode in connectorEdges do
                    if isParameterEdge startPin endPin then
                        let minimumY = startNode.Y + startNode.Height + 48.

                        if endNode.Y < minimumY then
                            endNode.Y <- minimumY

                settleColumnCollisions()

            let minX = nodes |> Array.map _.X |> Array.min
            let minY = nodes |> Array.map _.Y |> Array.min

            if minX < leftMargin then
                let shift = leftMargin - minX
                nodes |> Array.iter (fun node -> node.X <- node.X + shift)

            if minY < topMargin then
                let shift = topMargin - minY
                nodes |> Array.iter (fun node -> node.Y <- node.Y + shift)

            let right =
                nodes
                |> Array.map (fun node -> node.X + node.Width)
                |> Array.max

            let bottom =
                nodes
                |> Array.map (fun node -> node.Y + node.Height)
                |> Array.max

            let requiredWidth = right + leftMargin
            let requiredHeight = bottom + topMargin

            drawing.Width <- max drawing.Width requiredWidth
            drawing.Height <- max drawing.Height requiredHeight

            let shiftInsideCanvas () =
                let minX = nodes |> Array.map _.X |> Array.min
                let minY = nodes |> Array.map _.Y |> Array.min

                if minX < leftMargin then
                    let shift = leftMargin - minX
                    nodes |> Array.iter (fun node -> node.X <- node.X + shift)

                if minY < topMargin then
                    let shift = topMargin - minY
                    nodes |> Array.iter (fun node -> node.Y <- node.Y + shift)

                let overflowX =
                    nodes
                    |> Array.map (fun node -> node.X + node.Width)
                    |> Array.max
                    |> fun right -> right + leftMargin - drawing.Width

                let overflowY =
                    nodes
                    |> Array.map (fun node -> node.Y + node.Height)
                    |> Array.max
                    |> fun bottom -> bottom + topMargin - drawing.Height

                if overflowX > 0. then
                    drawing.Width <- drawing.Width + overflowX

                if overflowY > 0. then
                    drawing.Height <- drawing.Height + overflowY

            shiftInsideCanvas()

            nodes |> Array.iter _.ClampToDrawing()
            shiftInsideCanvas()
            this.MarkGraphDirty()

    member this.DeleteSelectedCommand =
        SimpleCommand((fun _ -> this.DeleteSelectedElement()), (fun _ -> not (isNull selectedNode)))
        :> ICommand

    member this.ArrangeGraphCommand =
        SimpleCommand((fun _ -> this.ArrangeGraph()), (fun _ -> true))
        :> ICommand

    member this.RunCommand =
        SimpleCommand(
            (fun _ ->
                match this.ValidateGraph() with
                | Ok () ->
                    this.SetRunInProgress(true)
                    async {
                        let! fileDirectoryInputsResolved = this.ResolveFileDirectoryInputs()

                        if fileDirectoryInputsResolved then
                            let generatedProgram = PipelineCodeGenerator.generateSavedGraph (this.ExportGraph())
                            this.AppendGeneratedProgram(generatedProgram)
                            do! this.BuildAndRunGeneratedProgram(generatedProgram)
                        else
                            this.SetRunInProgress(false)
                    }
                    |> Async.StartImmediate
                | Error message -> this.AppendGeneratedProgram(message)),
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

        let canonicalFunctionId functionId =
            match functionId with
            | "Plot" -> "Chart"
            | "SqrtFloat64" -> "UnaryImageFunction"
            | "MaxOfPair"
            | "MinOfPair" -> "ImageOpImage"
            | _ -> functionId

        let canonicalSavedNode (savedNode: SavedNode) =
            let pairOperation =
                match savedNode.FunctionId with
                | "MaxOfPair" -> Some "max"
                | "MinOfPair" -> Some "min"
                | _ -> None

            match pairOperation with
            | Some operation ->
                let operationParameter =
                    { Key = "operation"
                      Value = operation
                      UseInput = false }

                { savedNode with
                    FunctionId = "ImageOpImage"
                    Parameters = Array.append [| operationParameter |] savedNode.Parameters }
            | None ->
                { savedNode with FunctionId = canonicalFunctionId savedNode.FunctionId }

        let loadedNodes =
            savedGraph.Nodes
            |> Array.map (fun savedNode ->
                let savedNode = canonicalSavedNode savedNode
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
                            refreshNodePins)

                    watchState node.State
                    node.X <- savedNode.X
                    node.Y <- savedNode.Y
                    node.ClampToDrawing()
                    node.SyncMoveOrigin()
                    setParameterValues node.State savedNode.Parameters
                    drawing.Nodes.Add(node :> INode)
                    savedNode.Id, node)
            |> Map.ofArray

        for edge in savedGraph.Edges do
            match loadedNodes |> Map.tryFind edge.FromNode, loadedNodes |> Map.tryFind edge.ToNode with
            | Some fromNode, Some toNode ->
                let fromKind =
                    if String.IsNullOrWhiteSpace edge.FromKind then DataOutput else PipelinePinKind.ofString edge.FromKind

                let toKind =
                    if String.IsNullOrWhiteSpace edge.ToKind then DataInput else PipelinePinKind.ofString edge.ToKind

                let toPort =
                    if toKind = ParameterInput && toNode.State.Definition.Id = "Chart" && edge.ToPort = 0 then
                        1
                    else
                        edge.ToPort

                match pinByKindIndex fromKind edge.FromPort fromNode, pinByKindIndex toKind toPort toNode with
                | Some startPin, Some endPin when canConnectPins startPin endPin ->
                    addConnector startPin endPin
                | Some _, Some _ ->
                    invalidOp $"Saved edge has incompatible port types: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{toPort}]"
                | _ ->
                    invalidOp $"Saved edge refers to a missing port: {edge.FromNode}[{edge.FromPort}] -> {edge.ToNode}[{toPort}]"
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

    member _.SuggestedGraphFileName =
        currentGraphPath
        |> Option.map Path.GetFileName
        |> Option.defaultValue "pipeline.json"

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
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddElementAt(functionId: string, x: float, y: float) =
        let node = createNode drawing.Nodes.Count functionId
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.
        node.ClampToDrawing()
        node.SyncMoveOrigin()

        drawing.Nodes.Add(node :> INode)
        this.SelectedNode <- node
        this.MarkGraphDirty()

    member this.AddPaletteDragElementAt(functionId: string, x: float, y: float, isOutsideGraph: bool) =
        let node = createNode drawing.Nodes.Count functionId
        node.State.IsPaletteDragOutside <- isOutsideGraph
        node.X <- x - node.Width / 2.
        node.Y <- y - node.Height / 2.
        node.SyncMoveOrigin()

        drawing.Nodes.Add(node :> INode)
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

    member this.DeleteSelectedElement() =
        if not (isNull selectedNode) then
            let nodes = pipelineNodes () |> Seq.toArray
            let currentIndex =
                nodes
                |> Array.tryFindIndex (fun node -> Object.ReferenceEquals(node, selectedNode))
                |> Option.defaultValue 0

            let pinsToRemove = selectedNode.Pins |> Seq.toArray

            let connectorsToRemove =
                drawing.Connectors
                |> Seq.filter (fun connector ->
                    pinsToRemove
                    |> Array.exists (fun pin -> Object.ReferenceEquals(pin, connector.Start) || Object.ReferenceEquals(pin, connector.End)))
                |> Seq.toArray

            for connector in connectorsToRemove do
                drawing.Connectors.Remove(connector) |> ignore

            drawing.Nodes.Remove(selectedNode) |> ignore

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
        if not (isNull selectedNode) then
            let trashLeft = max 0. (drawing.Width - trashWidth - margin)
            let trashTop = max 0. (drawing.Height - trashHeight - margin)

            if selectedNode.X + selectedNode.Width >= trashLeft && selectedNode.Y + selectedNode.Height >= trashTop then
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
