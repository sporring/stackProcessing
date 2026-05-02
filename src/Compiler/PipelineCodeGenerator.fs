namespace Compiler

open System
open System.Collections.Generic
open System.Text
open Graph

module PipelineCodeGenerator =
    type private ParameterExpression =
        { Value: string
          IsLinked: bool }

    type private NamedBinding =
        { Name: string
          Dependencies: Set<string>
          Text: string }

    let private savedParamValue key (node: SavedNode) =
        node.Parameters
        |> Seq.tryFind (fun p -> p.Key = key)
        |> Option.map _.Value
        |> Option.defaultValue ""

    let private quote (value: string) =
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

    let private hasSuffix suffixes (value: string) =
        suffixes
        |> List.exists (fun suffix -> value.EndsWith(suffix, StringComparison.OrdinalIgnoreCase))

    let private float64Literal (value: string) =
        let trimmed = value.Trim()
        let lower = trimmed.ToLowerInvariant()

        if String.IsNullOrWhiteSpace trimmed
           || trimmed.Contains(".", StringComparison.Ordinal)
           || trimmed.Contains("e", StringComparison.OrdinalIgnoreCase)
           || lower = "nan"
           || lower = "infinity"
           || lower = "-infinity" then
            trimmed
        else
            $"{trimmed}.0"

    let private numericLiteral numericType (value: string) =
        let trimmed = value.Trim()

        match numericType with
        | UInt8 ->
            if hasSuffix [ "uy" ] trimmed then trimmed else $"{trimmed}uy"
        | Int8 ->
            if hasSuffix [ "y" ] trimmed then trimmed else $"{trimmed}y"
        | UInt16 ->
            if hasSuffix [ "us" ] trimmed then trimmed else $"{trimmed}us"
        | Int16 ->
            if hasSuffix [ "s" ] trimmed then trimmed else $"{trimmed}s"
        | UInt32 ->
            if hasSuffix [ "u" ] trimmed then trimmed else $"{trimmed}u"
        | Int32 ->
            trimmed
        | UInt64 ->
            if hasSuffix [ "ul" ] trimmed then trimmed else $"{trimmed}UL"
        | Int64 ->
            if hasSuffix [ "l" ] trimmed then trimmed else $"{trimmed}L"
        | Float32 ->
            if hasSuffix [ "f" ] trimmed then trimmed else $"{float64Literal trimmed}f"
        | Float64
        | Number ->
            float64Literal trimmed
        | Complex ->
            if trimmed.StartsWith("System.Numerics.Complex", StringComparison.Ordinal)
               || trimmed.StartsWith("Complex", StringComparison.Ordinal) then
                trimmed
            else
                $"System.Numerics.Complex({float64Literal trimmed}, 0.0)"

    let private literalValue basicType value =
        match basicType with
        | BasicType.Numeric numericType -> numericLiteral numericType value
        | BasicType.Bool -> value.Trim().ToLowerInvariant()
        | BasicType.String -> quote value
        | BasicType.Unit -> "()"

    let private scalarValueLiteral (node: SavedNode) =
        let value = savedParamValue "value" node

        let scalarType =
            savedParamValue "type" node
            |> BasicType.tryParse

        match scalarType with
        | Some basicType -> literalValue basicType value
        | _ -> value

    let private optionUInt (value: string) =
        let trimmed = value.Trim()
        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed.TrimEnd('u', 'U')}u)"

    let private optionRaw (value: string) =
        let trimmed = value.Trim()
        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed})"

    let private pixelTypeNameFromSuffix suffix =
        match suffix with
        | "UInt8" -> "uint8"
        | "Int8" -> "int8"
        | "UInt16" -> "uint16"
        | "Int16" -> "int16"
        | "UInt32" -> "uint32"
        | "Int32" -> "int32"
        | "UInt64" -> "uint64"
        | "Int64" -> "int64"
        | "Float32" -> "float32"
        | "Float64" -> "float"
        | "Complex" -> "System.Numerics.Complex"
        | _ -> suffix

    let private pixelTypeNameFromParameter key defaultType (node: SavedNode) =
        let configuredType = savedParamValue key node

        if String.IsNullOrWhiteSpace configuredType then
            pixelTypeNameFromSuffix defaultType
        else
            pixelTypeNameFromSuffix configuredType

    let private sourcePrefix availableMemory line =
        $"source {availableMemory}{Environment.NewLine}{line}"

    let private pairFunctionName functionId =
        match functionId with
        | "AddPair" -> Some "addPair"
        | "MulPair" -> Some "mulPair"
        | "DivPair" -> Some "divPair"
        | _ -> None

    let private scalarImageFunctionName (node: SavedNode) =
        let operation =
            match savedParamValue "operation" node with
            | "+"
            | "-"
            | "*"
            | "/" as operation -> operation
            | _ -> "*"

        match node.FunctionId, operation with
        | "ImageOpScalar", "+" -> Some "imageAddScalar"
        | "ImageOpScalar", "-" -> Some "imageSubScalar"
        | "ImageOpScalar", "*" -> Some "imageMulScalar"
        | "ImageOpScalar", "/" -> Some "imageDivScalar"
        | "ScalarOpImage", "+" -> Some "scalarAddImage"
        | "ScalarOpImage", "-" -> Some "scalarSubImage"
        | "ScalarOpImage", "*" -> Some "scalarMulImage"
        | "ScalarOpImage", "/" -> Some "scalarDivImage"
        | _ -> None

    let private isScalarImageFunction functionId =
        functionId = "ImageOpScalar" || functionId = "ScalarOpImage"

    let private scalarTypeName (node: SavedNode) =
        savedParamValue "type" node

    let private scalarNames (scalarNodes: SavedNode array) =
        scalarNodes
        |> Array.groupBy scalarTypeName
        |> Array.collect (fun (typeName, nodes) ->
            nodes
            |> Array.mapi (fun index node -> node.Id, $"{typeName}{index}"))
        |> Map.ofArray

    let private computeStatsFieldName portIndex =
        [| "NumPixels"; "Mean"; "Std"; "Min"; "Max"; "Sum"; "Var" |]
        |> Array.tryItem portIndex

    let private parameterExpression (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (node: SavedNode) parameterIndex key =
        let linkedExpression =
            graph.Edges
            |> Seq.tryFind (fun edge ->
                edge.ToNode = node.Id
                && edge.ToKind = "parameterInput"
                && edge.ToPort = parameterIndex)
            |> Option.bind (fun edge ->
                match edge.FromKind with
                | "scalarOutput" ->
                    scalarNamesByNodeId |> Map.tryFind edge.FromNode
                | "reducerOutput" ->
                    match statsNamesByNodeId |> Map.tryFind edge.FromNode, computeStatsFieldName edge.FromPort with
                    | Some statsName, Some fieldName -> Some $"{statsName}.{fieldName}"
                    | _ -> None
                | _ ->
                    None)

        match linkedExpression with
        | Some expression ->
            { Value = expression
              IsLinked = true }
        | None ->
            { Value = savedParamValue key node
              IsLinked = false }

    let private scalarBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let name = scalarNamesByNodeId |> Map.find node.Id

        let value =
            match node.FunctionId with
            | "ScalarOp" ->
                let parameterExpression key =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId node index key)
                    |> Option.defaultValue { Value = ""; IsLinked = false }

                let operation =
                    match savedParamValue "operation" node with
                    | "+"
                    | "-"
                    | "*"
                    | "/" as operation -> operation
                    | _ -> "*"

                let scalarType =
                    savedParamValue "type" node
                    |> BasicType.tryParse
                    |> Option.defaultValue (BasicType.Numeric Float64)

                let typedParameter key =
                    let expression = parameterExpression key
                    if expression.IsLinked then expression.Value else literalValue scalarType expression.Value

                let left = typedParameter "a"
                let right = typedParameter "b"
                $"({left} {operation} {right})"
            | _ ->
                scalarValueLiteral node

        name, $"let {name} = {value}"

    let private savedElementLine (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpression key =
            node.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId node index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let dynamicParameterType key =
            let configuredNumericType () =
                savedParamValue "type" node
                |> NumericType.tryParse
                |> Option.map BasicType.Numeric

            match node.FunctionId, key with
            | "ScalarOp", ("a" | "b") ->
                configuredNumericType ()
            | id, "value" when isScalarImageFunction id ->
                configuredNumericType ()
            | _ ->
                BuiltInCatalog.tryFind node.FunctionId
                |> Option.bind (fun definition ->
                    definition.Parameters
                    |> Seq.tryFind (fun parameter -> parameter.Key = key)
                    |> Option.map _.Type)

        let parameterValue key =
            let expression = parameterExpression key

            if expression.IsLinked then
                expression.Value
            else
                match dynamicParameterType key with
                | Some(BasicType.Numeric numericType) ->
                    numericLiteral numericType expression.Value
                | Some BasicType.Bool ->
                    expression.Value.Trim().ToLowerInvariant()
                | Some BasicType.Unit ->
                    "()"
                | Some BasicType.String
                | None ->
                    expression.Value

        let quotedParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        match node.FunctionId with
        | "Zero" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> zero<{pixelType}> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "ReadRandom" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let depth = parameterValue "depth"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readRandom<{pixelType}> {depth} {input} {suffix}" |> sourcePrefix availableMemory
        | "ReadChunks" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readChunks<{pixelType}> {input} {suffix}" |> sourcePrefix availableMemory
        | "Read" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> read<{pixelType}> {input} {suffix}" |> sourcePrefix availableMemory
        | "WriteInChunks" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> writeInChunks {output} {suffix} {chunkX} {chunkY} {chunkZ}"
        | "SqrtFloat64" ->
            ">=> sqrt"
        | id when isScalarImageFunction id ->
            let value = parameterValue "value"
            $">=> {scalarImageFunctionName node |> Option.get} {value}"
        | id when pairFunctionName id |> Option.isSome ->
            $">>=> {pairFunctionName id |> Option.get}"
        | "DiscreteGaussian" ->
            let sigma = parameterValue "sigma"
            let outputRegionMode = parameterValue "outputRegionMode" |> optionRaw
            let boundaryCondition = parameterValue "boundaryCondition" |> optionRaw
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> discreteGaussian {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | "ConvGauss" ->
            let sigma = parameterValue "sigma"
            $">=> convGauss {sigma}"
        | "FiniteDiff" ->
            let sigma = parameterValue "sigma"
            let axis1 = parameterValue "axis1"
            let axis2 = parameterValue "axis2"
            $">=> finiteDiff {sigma} {axis1} {axis2}"
        | "ComputeStats" ->
            ">=> computeStats ()"
        | "AddNormalNoise" ->
            let mean = parameterValue "mean"
            let std = parameterValue "std"
            $">=> addNormalNoise {mean} {std}"
        | "Threshold" ->
            let lower = parameterValue "lower"
            let upper = parameterValue "upper"
            $">=> threshold {lower} {upper}"
        | "Erode" ->
            let radius = parameterValue "radius"
            $">=> erode {radius}"
        | "Dilate" ->
            let radius = parameterValue "radius"
            $">=> dilate {radius}"
        | "Opening" ->
            let radius = parameterValue "radius"
            $">=> opening {radius}"
        | "Closing" ->
            let radius = parameterValue "radius"
            $">=> closing {radius}"
        | "ConnectedComponents" ->
            let windowSize = parameterValue "windowSize"
            $">=> connectedComponents {windowSize}"
        | "PermuteAxes" ->
            let axes = parameterValue "axes"
            let tileSize = parameterValue "tileSize"
            $">=> permuteAxes {axes} {tileSize}"
        | "Cast" ->
            let sourceType = pixelTypeNameFromParameter "sourceType" "Float64" node
            let targetType = pixelTypeNameFromParameter "targetType" "Float64" node
            $">=> cast<{sourceType},{targetType}>"
        | "Write" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> write {output} {suffix}"
        | _ ->
            $"// Unsupported element: {node.FunctionId}"

    let generateSavedGraph (graph: SavedGraph) =
        let builder = StringBuilder()
        let nodesById = graph.Nodes |> Seq.map (fun node -> node.Id, node) |> Map.ofSeq
        let scalarNodes =
            graph.Nodes
            |> Array.filter (fun node -> node.FunctionId = "Scalar" || node.FunctionId = "ScalarOp")

        let scalarNamesByNodeId = scalarNames scalarNodes
        let statsNodesWithLinkedFields =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter (fun node -> node.FunctionId = "ComputeStats")
            |> Array.distinctBy _.Id

        let statsNamesByNodeId =
            statsNodesWithLinkedFields
            |> Array.mapi (fun index node -> node.Id, $"ImageStats{index}")
            |> Map.ofArray

        let newLine = Environment.NewLine

        let dataEdges =
            graph.Edges
            |> Array.filter (fun edge -> edge.FromKind <> "scalarOutput" && edge.ToKind <> "parameterInput")

        let bindingNameForOutput edge =
            match edge.FromKind with
            | "scalarOutput" ->
                scalarNamesByNodeId |> Map.tryFind edge.FromNode
            | "reducerOutput" ->
                statsNamesByNodeId |> Map.tryFind edge.FromNode
            | _ ->
                None

        let parameterBindingDependencies (node: SavedNode) =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.ToNode = node.Id && edge.ToKind = "parameterInput" then
                    bindingNameForOutput edge
                else
                    None)
            |> Set.ofArray

        let rec pipelineBindingDependencies visited (node: SavedNode) =
            if visited |> Set.contains node.Id then
                Set.empty
            else
                let visited = visited |> Set.add node.Id
                let parameterDependencies = parameterBindingDependencies node

                let upstreamDependencies =
                    dataEdges
                    |> Array.filter (fun edge -> edge.ToNode = node.Id)
                    |> Array.choose (fun edge -> nodesById |> Map.tryFind edge.FromNode)
                    |> Array.map (pipelineBindingDependencies visited)
                    |> Set.unionMany

                Set.union parameterDependencies upstreamDependencies

        let indentBlock spaces (text: string) =
            let prefix = String.replicate spaces " "

            text.Split([| newLine |], StringSplitOptions.None)
            |> Array.map (fun line -> if String.IsNullOrWhiteSpace line then line else prefix + line)
            |> String.concat newLine

        let parenthesizeBlock (text: string) =
            $"({newLine}{indentBlock 4 text}{newLine})"

        let incomingDataEdge nodeId port =
            dataEdges
            |> Array.tryFind (fun edge -> edge.ToNode = nodeId && edge.ToPort = port)

        let sameOutputEdge (left: SavedEdge) (right: SavedEdge) =
            left.FromNode = right.FromNode
            && left.FromPort = right.FromPort
            && left.FromKind = right.FromKind

        let stageCall (node: SavedNode) =
            let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId node

            if line.StartsWith(">=> ", StringComparison.Ordinal) then
                Some(line.Substring(4))
            elif line.StartsWith(">>=> ", StringComparison.Ordinal) then
                Some(line.Substring(5))
            else
                None

        let composeStages (left: string) (right: string) =
            $"{left}{newLine}>=> {right}"

        let formatStageTuple (left: string) (right: string) =
            if left.Contains(newLine, StringComparison.Ordinal) || right.Contains(newLine, StringComparison.Ordinal) then
                $">=>> ({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine})"
            else
                $">=>> ({left}, {right})"

        let rec pipelineExpression visited (node: SavedNode) =
            if visited |> Set.contains node.Id then
                $"// Cannot generate F#: cycle detected at {node.FunctionId}"
            else
                let visited = visited |> Set.add node.Id

                let inputExpression port =
                    match incomingDataEdge node.Id port with
                    | Some edge ->
                        match nodesById |> Map.tryFind edge.FromNode with
                        | Some upstream -> pipelineExpression visited upstream
                        | None -> $"// Cannot generate F#: missing upstream node {edge.FromNode}"
                    | None -> $"// Cannot generate F#: missing input {port} for {node.FunctionId}"

                let rec branchStageExpression commonInputEdge visited (node: SavedNode) =
                    if visited |> Set.contains node.Id then
                        None
                    else
                        let visited = visited |> Set.add node.Id

                        match incomingDataEdge node.Id 0, stageCall node with
                        | Some incoming, Some call when sameOutputEdge incoming commonInputEdge ->
                            Some call
                        | Some incoming, Some call ->
                            nodesById
                            |> Map.tryFind incoming.FromNode
                            |> Option.bind (branchStageExpression commonInputEdge visited)
                            |> Option.map (fun upstream -> composeStages upstream call)
                        | _ ->
                            None

                let sharedFanOutExpression () =
                    match incomingDataEdge node.Id 0, incomingDataEdge node.Id 1 with
                    | Some leftEdge, Some rightEdge when sameOutputEdge leftEdge rightEdge ->
                        nodesById
                        |> Map.tryFind leftEdge.FromNode
                        |> Option.bind (fun sharedNode ->
                            let shared = pipelineExpression visited sharedNode
                            let fanOut = formatStageTuple "idStage \"left\"" "idStage \"right\""
                            pairFunctionName node.FunctionId
                            |> Option.map (fun pairFunction -> $"{shared}{newLine}{fanOut}{newLine}>>=> {pairFunction}"))
                    | Some leftEdge, Some rightEdge ->
                        match nodesById |> Map.tryFind leftEdge.FromNode, nodesById |> Map.tryFind rightEdge.FromNode with
                        | Some leftNode, Some rightNode ->
                            match incomingDataEdge leftNode.Id 0, incomingDataEdge rightNode.Id 0 with
                            | Some leftInput, Some rightInput when sameOutputEdge leftInput rightInput ->
                                match nodesById |> Map.tryFind leftInput.FromNode with
                                | Some sharedNode ->
                                    match branchStageExpression leftInput Set.empty leftNode, branchStageExpression leftInput Set.empty rightNode with
                                    | Some leftStage, Some rightStage ->
                                        let shared = pipelineExpression visited sharedNode
                                        pairFunctionName node.FunctionId
                                        |> Option.map (fun pairFunction -> $"{shared}{newLine}{formatStageTuple leftStage rightStage}{newLine}>>=> {pairFunction}")
                                    | _ -> None
                                | None -> None
                            | _ -> None
                        | _ -> None
                    | _ -> None

                match node.FunctionId with
                | id when pairFunctionName id |> Option.isSome ->
                    match sharedFanOutExpression () with
                    | Some expression ->
                        expression
                    | None ->
                        let pairFunction = pairFunctionName id |> Option.get
                        let left = inputExpression 0
                        let right = inputExpression 1
                        let left = parenthesizeBlock left
                        let right = parenthesizeBlock right
                        $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}>>=> {pairFunction}"
                | _ ->
                    let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId node

                    match incomingDataEdge node.Id 0 with
                    | Some _ ->
                        let input = inputExpression 0
                        $"{input}{newLine}{line}"
                    | None ->
                        line

        let generatedPipelines =
            let appendSinkIfTerminalWrite (node: SavedNode) expression =
                match node.FunctionId with
                | "Write"
                | "WriteInChunks" ->
                    $"{expression}{newLine}|> sink"
                | "ComputeStats" ->
                    $"{expression}{newLine}|> drain"
                | _ ->
                    expression

            let terminalNodes =
                graph.Nodes
                |> Array.filter (fun node ->
                    node.FunctionId <> "Scalar"
                    && node.FunctionId <> "ScalarOp"
                    && not (statsNamesByNodeId |> Map.containsKey node.Id)
                    && not (dataEdges |> Array.exists (fun edge -> edge.FromNode = node.Id)))

            terminalNodes
            |> Array.map (fun node -> pipelineExpression Set.empty node |> appendSinkIfTerminalWrite node)

        let orderedBindings () =
            let statsBindings =
                statsNodesWithLinkedFields
                |> Array.map (fun node ->
                    let name = statsNamesByNodeId |> Map.find node.Id
                    let expression = pipelineExpression Set.empty node
                    let body = indentBlock 4 $"{expression}{newLine}|> drain"

                    { Name = name
                      Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                      Text = $"let {name} ={newLine}{body}" })

            let scalarBindings =
                scalarNodes
                |> Array.map (fun node ->
                    let name, text = scalarBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId node

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = text })

            let bindings = Array.append scalarBindings statsBindings
            let bindingsByName = bindings |> Array.map (fun binding -> binding.Name, binding) |> Map.ofArray
            let visited = HashSet<string>()
            let ordered = ResizeArray<NamedBinding>()

            let rec visit (binding: NamedBinding) =
                if visited.Add binding.Name then
                    binding.Dependencies
                    |> Seq.choose (fun dependency -> bindingsByName |> Map.tryFind dependency)
                    |> Seq.iter visit

                    ordered.Add binding

            bindings |> Array.iter visit
            ordered |> Seq.toArray

        builder.AppendLine("open StackProcessing") |> ignore
        builder.AppendLine() |> ignore

        let bindings = orderedBindings ()

        if bindings.Length > 0 then
            bindings
            |> Array.iter (fun binding -> builder.AppendLine(binding.Text) |> ignore)
            builder.AppendLine() |> ignore

        generatedPipelines
        |> Seq.iteri (fun index expression ->
            if index > 0 then
                builder.AppendLine() |> ignore

            builder.AppendLine(expression) |> ignore)

        builder.ToString().TrimEnd()
