namespace Studio.Services

open System
open System.Text
open Graph
open Studio.Models

module PipelineCodeGenerator =
    type private ParameterExpression =
        { Value: string
          IsLinked: bool }

    let private paramValue key (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun p -> p.Key = key)
        |> Option.map _.Value
        |> Option.defaultValue ""

    let private savedParamValue key (node: SavedNode) =
        node.Parameters
        |> Seq.tryFind (fun p -> p.Key = key)
        |> Option.map _.Value
        |> Option.defaultValue ""

    let private quote (value: string) =
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

    let private scalarValueLiteral (node: SavedNode) =
        let value = savedParamValue "value" node

        match BuiltInCatalog.tryFind node.FunctionId with
        | Some definition ->
            match definition.Outputs |> List.tryHead |> Option.map _.Type with
            | Some(Scalar BasicType.String) -> quote value
            | _ -> value
        | None -> value

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

    let private elementLine (state: PipelineNodeState) =
        match state.Definition.Id with
        | "Source" ->
            let availableMemory = paramValue "availableMemory" state
            $"source {availableMemory}"
        | id when id.StartsWith("Read", StringComparison.Ordinal) ->
            let pixelType = paramValue "NumericType" state
            let input = paramValue "input" state |> quote
            let suffix = paramValue "suffix" state |> quote
            $"|> read<{pixelType}> {input} {suffix}"
        | "DiscreteGaussian" ->
            let sigma = paramValue "sigma" state
            let outputRegionMode = paramValue "outputRegionMode" state |> optionRaw
            let boundaryCondition = paramValue "boundaryCondition" state |> optionRaw
            let windowSize = paramValue "windowSize" state |> optionUInt
            $">=> discreteGaussian {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | id when id.StartsWith("Cast", StringComparison.Ordinal) ->
            let sourceType = paramValue "sourceType" state
            let targetType = paramValue "targetType" state
            $">=> cast<{sourceType},{targetType}>"
        | "Write" ->
            let output = paramValue "output" state |> quote
            let suffix = paramValue "suffix" state |> quote
            $">=> write {output} {suffix}"
        | "Sink" ->
            "|> sink"
        | id ->
            $"// Unsupported element: {state.Title}"

    let private scalarTypeName (node: SavedNode) =
        node.FunctionId.Substring("Scalar".Length)

    let private scalarNames (scalarNodes: SavedNode array) =
        scalarNodes
        |> Array.groupBy scalarTypeName
        |> Array.collect (fun (typeName, nodes) ->
            nodes
            |> Array.mapi (fun index node -> node.Id, $"{typeName}{index}"))
        |> Map.ofArray

    let private scalarBinding (namesByNodeId: Map<string, string>) (node: SavedNode) =
        let name = namesByNodeId |> Map.find node.Id
        let value = scalarValueLiteral node
        name, $"let {name} = {value}"

    let private parameterExpression (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (node: SavedNode) parameterIndex key =
        let fromLinkedScalar =
            graph.Edges
            |> Seq.tryFind (fun edge ->
                edge.ToNode = node.Id
                && edge.ToKind = "parameterInput"
                && edge.ToPort = parameterIndex)
            |> Option.bind (fun edge -> scalarNamesByNodeId |> Map.tryFind edge.FromNode)

        match fromLinkedScalar with
        | Some scalarName ->
            { Value = scalarName
              IsLinked = true }
        | None ->
            { Value = savedParamValue key node
              IsLinked = false }

    let private savedElementLine (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpression key =
            node.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId node index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let parameterValue key =
            (parameterExpression key).Value

        let quotedParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        match node.FunctionId with
        | "Source" ->
            let availableMemory = parameterValue "availableMemory"
            $"source {availableMemory}"
        | id when id.StartsWith("Read", StringComparison.Ordinal) ->
            let pixelType = parameterValue "NumericType"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> read<{pixelType}> {input} {suffix}"
        | "DiscreteGaussian" ->
            let sigma = parameterValue "sigma"
            let outputRegionMode = parameterValue "outputRegionMode" |> optionRaw
            let boundaryCondition = parameterValue "boundaryCondition" |> optionRaw
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> discreteGaussian {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | id when id.StartsWith("Cast", StringComparison.Ordinal) ->
            let sourceType = parameterValue "sourceType"
            let targetType = parameterValue "targetType"
            $">=> cast<{sourceType},{targetType}>"
        | "Write" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> write {output} {suffix}"
        | "Sink" ->
            "|> sink"
        | _ ->
            $"// Unsupported element: {node.FunctionId}"

    let generate (states: PipelineNodeState seq) =
        let builder = StringBuilder()
        builder.AppendLine("open StackProcessing") |> ignore
        builder.AppendLine() |> ignore

        states
        |> Seq.map elementLine
        |> Seq.iter (fun line -> builder.AppendLine(line) |> ignore)

        builder.ToString().TrimEnd()

    let generateSavedGraph (graph: SavedGraph) =
        let builder = StringBuilder()
        let nodesById = graph.Nodes |> Seq.map (fun node -> node.Id, node) |> Map.ofSeq
        let scalarNodes = graph.Nodes |> Array.filter (fun node -> node.FunctionId.StartsWith("Scalar", StringComparison.Ordinal))
        let scalarNamesByNodeId = scalarNames scalarNodes

        let dataEdges =
            graph.Edges
            |> Array.filter (fun edge -> edge.FromKind <> "scalarOutput" && edge.ToKind <> "parameterInput")

        let nextDataNodeId nodeId =
            dataEdges
            |> Array.tryFind (fun edge -> edge.FromNode = nodeId)
            |> Option.map _.ToNode

        let orderedPipelineNodes =
            let rec walk visited nodeId =
                if Set.contains nodeId visited then
                    []
                else
                    match nodesById |> Map.tryFind nodeId with
                    | Some node when not (node.FunctionId.StartsWith("Scalar", StringComparison.Ordinal)) ->
                        node :: (nextDataNodeId nodeId |> Option.map (walk (Set.add nodeId visited)) |> Option.defaultValue [])
                    | _ -> []

            let sourceNodes =
                graph.Nodes
                |> Array.filter (fun node -> node.FunctionId = "Source")

            let walked =
                sourceNodes
                |> Array.toList
                |> List.collect (fun source -> walk Set.empty source.Id)

            if walked.Length > 0 then
                walked
            else
                graph.Nodes
                |> Array.filter (fun node -> not (node.FunctionId.StartsWith("Scalar", StringComparison.Ordinal)))
                |> Array.toList

        builder.AppendLine("open StackProcessing") |> ignore
        builder.AppendLine() |> ignore

        if scalarNodes.Length > 0 then
            scalarNodes
            |> Array.map (scalarBinding scalarNamesByNodeId)
            |> Array.map snd
            |> Array.iter (fun line -> builder.AppendLine(line) |> ignore)

            builder.AppendLine() |> ignore

        orderedPipelineNodes
        |> Seq.map (savedElementLine graph nodesById scalarNamesByNodeId)
        |> Seq.iter (fun line -> builder.AppendLine(line) |> ignore)

        builder.ToString().TrimEnd()
