namespace Studio.Compiler

open System
open System.Collections.Generic
open System.Text
open System.Text.RegularExpressions
open Studio.Graph

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

    let private standardNumericConstant (value: string) =
        match value.Trim().ToLowerInvariant() with
        | "e" -> Some "System.Math.E"
        | "pi" -> Some "System.Math.PI"
        | _ -> None

    let private numericLiteral numericType (value: string) =
        let trimmed = value.Trim()

        match standardNumericConstant trimmed with
        | Some constant ->
            match numericType with
            | UInt8 -> $"uint8 {constant}"
            | Int8 -> $"int8 {constant}"
            | UInt16 -> $"uint16 {constant}"
            | Int16 -> $"int16 {constant}"
            | UInt32 -> $"uint32 {constant}"
            | Int32 -> $"int {constant}"
            | UInt64 -> $"uint64 {constant}"
            | Int64 -> $"int64 {constant}"
            | Float32 -> $"float32 {constant}"
            | Float64
            | Number -> constant
            | Complex -> $"System.Numerics.Complex({constant}, 0.0)"
        | None ->
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
        | BasicType.Map -> value.Trim()
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

    let private optionQualified moduleName (value: string) =
        let trimmed = value.Trim()

        if System.String.IsNullOrWhiteSpace trimmed || System.String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        else
            let inner =
                if trimmed.StartsWith("(Some ", StringComparison.Ordinal) && trimmed.EndsWith(")", StringComparison.Ordinal) then
                    trimmed.Substring(6, trimmed.Length - 7).Trim()
                elif trimmed.StartsWith("Some ", StringComparison.Ordinal) then
                    trimmed.Substring(5).Trim()
                else
                    trimmed

            let qualified =
                if inner.Contains(".", StringComparison.Ordinal) then inner else $"{moduleName}.{inner}"

            $"(Some {qualified})"

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

    let private uint64Literal (value: string) =
        let trimmed = value.Trim()

        if trimmed.EndsWith("UL", StringComparison.OrdinalIgnoreCase) then
            trimmed
        else
            $"{trimmed}UL"

    let private sourcePrefix availableMemory line =
        $"debug 1u {uint64Literal availableMemory}{Environment.NewLine}{line}"

    let private imageOpImageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "+" -> "addPair"
        | "-" -> "subPair"
        | "/" -> "divPair"
        | "max" -> "maxOfPair"
        | "min" -> "minOfPair"
        | _ -> "mulPair"

    let private pairStageFunctionName (node: SavedNode) =
        match node.FunctionId with
        | "ImageOpImage" -> Some(imageOpImageFunctionName node)
        | "ComplexFromReIm" -> Some "toComplex"
        | "ComplexPolar" -> Some "polarToComplex"
        | "ToVectorImage" -> Some "toVectorImage<float>"
        | "AppendVectorElement" -> Some "appendVectorElement"
        | "VectorDot" -> Some "vectorDot"
        | "VectorCross3D" -> Some "vectorCross3D"
        | "AffineRegistration" ->
            let maxIterations = savedParamValue "maxIterations" node |> numericLiteral Int32
            let initialLinearStep = savedParamValue "initialLinearStep" node |> numericLiteral Float64
            let initialTranslationStep = savedParamValue "initialTranslationStep" node |> numericLiteral Float64
            let minStep = savedParamValue "minStep" node |> numericLiteral Float64
            let stepShrink = savedParamValue "stepShrink" node |> numericLiteral Float64
            Some $"affineRegistrationMatrices {{ defaultAffineRegistrationOptions with MaxIterations = {maxIterations}; InitialLinearStep = {initialLinearStep}; InitialTranslationStep = {initialTranslationStep}; MinStep = {minStep}; StepShrink = {stepShrink} }}"
        | "ImageComparison" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let comparison =
                match savedParamValue "operation" node with
                | "=" | "==" | "equal" -> "equal"
                | "<>" | "!=" | "notEqual" -> "notEqual"
                | ">=" | "greaterEqual" -> "greaterEqual"
                | "<" | "less" -> "less"
                | "<=" | "lessEqual" -> "lessEqual"
                | _ -> "greater"
            Some $"{comparison}<{pixelType}>"
        | "MaskLogic" ->
            let logic =
                match savedParamValue "operation" node with
                | "or" | "||" -> "maskOr"
                | "xor" | "^" -> "maskXor"
                | _ -> "maskAnd"
            Some logic
        | _ -> None

    let private pairCompositionOperator (node: SavedNode) =
        match node.FunctionId with
        | "ImageOpImage" -> ">>=>"
        | _ -> ">=>"

    let private safeIdentifier (value: string) =
        let chars =
            value
            |> Seq.map (fun c -> if Char.IsLetterOrDigit c then c else '_')
            |> Seq.toArray

        new String(chars)

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

    let private unaryImageFunctionNames =
        [ "abs"; "acos"; "asin"; "atan"; "cos"; "sin"; "tan"; "exp"; "log10"; "log"; "round"; "sqrt"; "square" ]

    let private unaryImageFunctionName (node: SavedNode) =
        let configuredFunction = savedParamValue "function" node

        if unaryImageFunctionNames |> List.contains configuredFunction then
            configuredFunction
        else
            "sqrt"

    let private scalarFunctionExpression functionName argument =
        match functionName with
        | "abs" -> $"(System.Math.Abs {argument})"
        | "acos" -> $"(System.Math.Acos {argument})"
        | "asin" -> $"(System.Math.Asin {argument})"
        | "atan" -> $"(System.Math.Atan {argument})"
        | "cos" -> $"(System.Math.Cos {argument})"
        | "sin" -> $"(System.Math.Sin {argument})"
        | "tan" -> $"(System.Math.Tan {argument})"
        | "exp" -> $"(System.Math.Exp {argument})"
        | "log10" -> $"(System.Math.Log10 {argument})"
        | "log" -> $"(System.Math.Log {argument})"
        | "round" -> $"(System.Math.Round {argument})"
        | "square" -> $"({argument} * {argument})"
        | _ -> $"(System.Math.Sqrt {argument})"

    let private comparisonStageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "=" | "==" | "equal" -> "equal"
        | "<>" | "!=" | "notEqual" -> "notEqual"
        | ">=" | "greaterEqual" -> "greaterEqual"
        | "<" | "less" -> "less"
        | "<=" | "lessEqual" -> "lessEqual"
        | _ -> "greater"

    let private maskLogicStageFunctionName (node: SavedNode) =
        match savedParamValue "operation" node with
        | "or" | "||" -> "maskOr"
        | "xor" | "^" -> "maskXor"
        | _ -> "maskAnd"

    let private scalarTypeName (node: SavedNode) =
        match node.FunctionId with
        | "ScalarFunction" -> "Float64"
        | "OtsuThresholdFromHistogram"
        | "MomentsThresholdFromHistogram" -> "Float64"
        | _ -> savedParamValue "type" node

    let private scalarParameter key value =
        { Key = key
          Value = value
          UseInput = false }

    let private replaceFileDirectorySelectors (graph: SavedGraph) =
        let nodes =
            graph.Nodes
            |> Array.map (fun node ->
                if node.FunctionId = "FileDirectory" then
                    { node with
                        FunctionId = "Scalar"
                        Parameters =
                            [| scalarParameter "type" "String"
                               scalarParameter "value" (savedParamValue "value" node) |] }
                else
                    node)

        { graph with Nodes = nodes }

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

    let private isSingleValueReducerNode (node: SavedNode) =
        node.FunctionId = "SurfaceArea"
        || node.FunctionId = "Volume"
        || node.FunctionId = "PointPairDistances"
        || node.FunctionId = "FitBiasModel"
        || node.FunctionId = "FitBiasModelMasked"

    let private stackInfoFieldExpression bindingName portIndex =
        [| $"{bindingName}.dimensions"
           $"{bindingName}.size"
           $"{bindingName}.componentType"
           $"{bindingName}.numberOfComponents"
           $"{bindingName}.size[0]"
           $"{bindingName}.size[1]"
           $"{bindingName}.size[2]" |]
        |> Array.tryItem portIndex

    let private chunkInfoFieldExpression bindingName portIndex =
        [| $"{bindingName}.chunks"
           $"{bindingName}.size"
           $"{bindingName}.topLeftInfo.componentType"
           $"{bindingName}.topLeftInfo.numberOfComponents"
           $"{bindingName}.chunks[0]"
           $"{bindingName}.chunks[1]"
           $"{bindingName}.chunks[2]"
           $"{bindingName}.size[0]"
           $"{bindingName}.size[1]"
           $"{bindingName}.size[2]" |]
        |> Array.tryItem portIndex

    let private isTranslationTableNode (node: SavedNode) =
        node.FunctionId = "ComponentTranslationTable"

    let private isHistogramDataNode (node: SavedNode) =
        node.FunctionId = "HistogramData" || node.FunctionId = "EstimateHistogram"

    let private isQuantilesNode (node: SavedNode) =
        node.FunctionId = "Quantiles"

    let private isStackInfoNode (node: SavedNode) =
        node.FunctionId = "GetStackInfo"

    let private isChunkInfoNode (node: SavedNode) =
        node.FunctionId = "GetChunkInfo" || node.FunctionId = "GetZarrInfo" || node.FunctionId = "GetNexusInfo"

    let private quantileFieldExpression bindingName portIndex =
        if portIndex >= 0 && portIndex < 5 then Some $"{bindingName}[{portIndex}]" else None

    let private histogramFieldExpression (nodesById: Map<string, SavedNode>) nodeId bindingName portIndex =
        match nodesById |> Map.tryFind nodeId with
        | Some node when node.FunctionId = "EstimateHistogram" ->
            match portIndex with
            | 0 -> Some $"{bindingName}.Histogram"
            | 1 -> Some $"{bindingName}.Samples"
            | 2 -> Some $"{bindingName}.CdfHalfWidth"
            | 3 -> Some $"{bindingName}.HoldoutMaxCdfDelta"
            | _ -> None
        | _ ->
            if portIndex = 0 then Some bindingName else None

    let private parameterExpression (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (stackInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (node: SavedNode) parameterIndex key =
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
                    match statsNamesByNodeId |> Map.tryFind edge.FromNode with
                    | Some statsName ->
                        match nodesById |> Map.tryFind edge.FromNode with
                        | Some reducerNode when isSingleValueReducerNode reducerNode && edge.FromPort = 0 -> Some statsName
                        | Some reducerNode when reducerNode.FunctionId = "ComputeStats" ->
                            computeStatsFieldName edge.FromPort |> Option.map (fun fieldName -> $"{statsName}.{fieldName}")
                        | _ -> None
                    | _ ->
                        match translationTableNamesByNodeId |> Map.tryFind edge.FromNode with
                        | Some name -> Some name
                        | None ->
                            match histogramNamesByNodeId |> Map.tryFind edge.FromNode with
                            | Some name -> histogramFieldExpression nodesById edge.FromNode name edge.FromPort
                            | None ->
                                match quantileNamesByNodeId |> Map.tryFind edge.FromNode with
                                | Some name -> quantileFieldExpression name edge.FromPort
                                | None ->
                                    stackInfoNamesByNodeId
                                    |> Map.tryFind edge.FromNode
                                    |> Option.bind (fun name -> stackInfoFieldExpression name edge.FromPort)
                                    |> Option.orElseWith (fun () ->
                                        chunkInfoNamesByNodeId
                                        |> Map.tryFind edge.FromNode
                                        |> Option.bind (fun name -> chunkInfoFieldExpression name edge.FromPort))
                | _ ->
                    None)

        match linkedExpression with
        | Some expression ->
            { Value = expression
              IsLinked = true }
        | None ->
            { Value = savedParamValue key node
              IsLinked = false }

    let private scalarBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (stackInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let name = scalarNamesByNodeId |> Map.find node.Id

        let value =
            match node.FunctionId with
            | "ScalarOp" ->
                let parameterExpression key =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index key)
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
            | "ScalarFunction" ->
                let parameterExpression =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = "a")
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index "a")
                    |> Option.defaultValue { Value = ""; IsLinked = false }

                let argument =
                    if parameterExpression.IsLinked then
                        parameterExpression.Value
                    else
                        literalValue (BasicType.Numeric Float64) parameterExpression.Value

                scalarFunctionExpression (unaryImageFunctionName node) argument
            | "OtsuThresholdFromHistogram"
            | "MomentsThresholdFromHistogram" ->
                let functionName =
                    if node.FunctionId = "OtsuThresholdFromHistogram" then
                        "otsuThresholdFromHistogram"
                    else
                        "momentsThresholdFromHistogram"

                let histogram =
                    node.Parameters
                    |> Seq.tryFindIndex (fun parameter -> parameter.Key = "histogram")
                    |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index "histogram")
                    |> Option.defaultValue { Value = savedParamValue "histogram" node; IsLinked = false }

                $"{functionName} {histogram.Value}"
            | _ ->
                scalarValueLiteral node

        name, $"let {name} = {value}"

    let private stackInfoBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (stackInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpression key =
            node.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let stringArgument key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        let name = stackInfoNamesByNodeId |> Map.find node.Id
        let input = stringArgument "input"
        let suffix = stringArgument "suffix"
        name, $"let {name} = getStackInfo {input} {suffix}"

    let private chunkInfoBinding (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (stackInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpression key =
            node.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let stringArgument key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        let name = chunkInfoNamesByNodeId |> Map.find node.Id
        let input = stringArgument "input"

        if node.FunctionId = "GetZarrInfo" then
            let multiscaleIndex = parameterExpression "multiscaleIndex"
            let datasetIndex = parameterExpression "datasetIndex"
            name, $"let {name} = getZarrInfo {input} {multiscaleIndex.Value} {datasetIndex.Value}"
        elif node.FunctionId = "GetNexusInfo" then
            let datasetPath = stringArgument "datasetPath"
            let frameAxis = parameterExpression "frameAxis"
            let yAxis = parameterExpression "yAxis"
            let xAxis = parameterExpression "xAxis"
            name, $"let {name} = getNexusInfo {input} {datasetPath} {frameAxis.Value} {yAxis.Value} {xAxis.Value}"
        else
            let suffix = stringArgument "suffix"
            name, $"let {name} = getChunkInfo {input} {suffix}"

    let private savedElementLine (graph: SavedGraph) (nodesById: Map<string, SavedNode>) (scalarNamesByNodeId: Map<string, string>) (statsNamesByNodeId: Map<string, string>) (translationTableNamesByNodeId: Map<string, string>) (histogramNamesByNodeId: Map<string, string>) (quantileNamesByNodeId: Map<string, string>) (stackInfoNamesByNodeId: Map<string, string>) (chunkInfoNamesByNodeId: Map<string, string>) (node: SavedNode) =
        let parameterExpressionForNode (targetNode: SavedNode) key =
            targetNode.Parameters
            |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
            |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId targetNode index key)
            |> Option.defaultValue { Value = ""; IsLinked = false }

        let parameterExpression key =
            parameterExpressionForNode node key

        let dynamicParameterTypeForNode (targetNode: SavedNode) key =
            let configuredNumericType () =
                savedParamValue "type" targetNode
                |> NumericType.tryParse
                |> Option.map BasicType.Numeric

            match targetNode.FunctionId, key with
            | "ScalarOp", ("a" | "b") ->
                configuredNumericType ()
            | "ScalarFunction", "a" ->
                Some(BasicType.Numeric Float64)
            | id, "value" when isScalarImageFunction id ->
                configuredNumericType ()
            | _ ->
                BuiltInCatalog.tryFind targetNode.FunctionId
                |> Option.bind (fun definition ->
                    definition.Parameters
                    |> Seq.tryFind (fun parameter -> parameter.Key = key)
                    |> Option.map _.Type)

        let parameterValueForNode targetNode key =
            let expression = parameterExpressionForNode targetNode key

            if expression.IsLinked then
                expression.Value
            else
                match dynamicParameterTypeForNode targetNode key with
                | Some(BasicType.Numeric numericType) ->
                    numericLiteral numericType expression.Value
                | Some BasicType.Bool ->
                    expression.Value.Trim().ToLowerInvariant()
                | Some BasicType.Unit ->
                    "()"
                | Some BasicType.Map ->
                    expression.Value
                | Some BasicType.String
                | None ->
                    expression.Value

        let parameterValue key =
            parameterValueForNode node key

        let rec inferredImagePixelType visited (targetNode: SavedNode) inputPort =
            if visited |> Set.contains targetNode.Id then
                None
            else
                let visited = visited |> Set.add targetNode.Id

                graph.Edges
                |> Array.tryFind (fun edge ->
                    edge.ToNode = targetNode.Id
                    && edge.ToKind <> "parameterInput"
                    && edge.ToPort = inputPort)
                |> Option.bind (fun edge ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.bind (fun sourceNode ->
                        match sourceNode.FunctionId, edge.FromPort with
                        | "SerialEstTrans", (0 | 1) ->
                            inferredImagePixelType visited sourceNode 0
                        | "Cast", 0 ->
                            Some(pixelTypeNameFromParameter "targetType" "Float64" sourceNode)
                        | _ ->
                            let configured = savedParamValue "type" sourceNode
                            if String.IsNullOrWhiteSpace configured then None else Some(pixelTypeNameFromSuffix configured)))

        let pipelinePixelType defaultType =
            inferredImagePixelType Set.empty node 0
            |> Option.defaultValue (pixelTypeNameFromParameter "type" defaultType node)

        let quotedParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then expression.Value else quote expression.Value

        let stringParameter key =
            let expression = parameterExpression key
            if expression.IsLinked then $"(string {expression.Value})" else quote expression.Value

        let printInputNameForNode (printNode: SavedNode) key index =
            graph.Edges
            |> Seq.tryFind (fun edge ->
                edge.ToNode = printNode.Id
                && edge.ToKind = "parameterInput"
                && edge.ToPort = index)
            |> Option.bind (fun edge ->
                match edge.FromKind with
                | "reducerOutput" ->
                    computeStatsFieldName edge.FromPort
                    |> Option.orElseWith (fun () ->
                        nodesById
                        |> Map.tryFind edge.FromNode
                        |> Option.bind (fun sourceNode ->
                            BuiltInCatalog.tryFind sourceNode.FunctionId
                            |> Option.bind (fun definition ->
                                definition.Outputs
                                |> List.tryItem edge.FromPort
                                |> Option.map _.Name)))
                | "scalarOutput" ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.map (fun sourceNode ->
                            if sourceNode.FunctionId = "Tap" then
                                "I"
                            elif sourceNode.FunctionId = "Scalar" then
                                savedParamValue "type" sourceNode
                            else
                                sourceNode.FunctionId)
                | _ ->
                    None)
            |> Option.orElseWith (fun () ->
                printNode.Parameters
                |> Seq.tryFind (fun parameter -> parameter.Key = key)
                |> Option.map _.Value)
            |> Option.map (fun name ->
                let index = name.IndexOf(':')

                if index > 0 then
                    name.Substring(0, index).Trim()
                else
                    name.Trim())
            |> Option.filter (String.IsNullOrWhiteSpace >> not)
            |> Option.defaultValue key

        let printInputName key index =
            printInputNameForNode node key index

        let interpolatedStringExpression (format: string) (inputs: (string * string * string) list) =
            let byName =
                inputs
                |> List.collect (fun (key, name, expression) -> [ key, expression; name, expression ])
                |> Map.ofList

            let escapeText (value: string) =
                value
                    .Replace("\\\\", "\u0000")
                    .Replace("\\n", "\n")
                    .Replace("\\r", "\r")
                    .Replace("\\t", "\t")
                    .Replace("\\\"", "\"")
                    .Replace("\u0000", "\\")
                    .Replace("\\", "\\\\")
                    .Replace("\"", "\\\"")
                    .Replace("{", "{{")
                    .Replace("}", "}}")

            let builder = StringBuilder()
            let mutable offset = 0

            for m in Regex.Matches(format, @"\{([^{}]+)\}") do
                if m.Index > offset then
                    builder.Append(escapeText (format.Substring(offset, m.Index - offset))) |> ignore

                let name = m.Groups[1].Value.Trim()

                match byName |> Map.tryFind name with
                | Some expression ->
                    builder.Append("{").Append(expression).Append("}") |> ignore
                | None ->
                    builder.Append(escapeText m.Value) |> ignore

                offset <- m.Index + m.Length

            if offset < format.Length then
                builder.Append(escapeText (format.Substring offset)) |> ignore

            "$\"" + builder.ToString() + "\""

        match node.FunctionId with
        | "Zero" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> zero<{pixelType}> {width} {height} {depth}" |> sourcePrefix availableMemory
        | "CoordinateX" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateX {width} {height} {depth}" |> sourcePrefix availableMemory
        | "CoordinateY" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateY {width} {height} {depth}" |> sourcePrefix availableMemory
        | "CoordinateZ" ->
            let availableMemory = parameterValue "availableMemory"
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            $"|> coordinateZ {width} {height} {depth}" |> sourcePrefix availableMemory
        | "NormalNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let mean = parameterValue "mean"
            let std = parameterValue "std"
            $"|> normalNoise<{pixelType}> {width} {height} {depth} {mean} {std}" |> sourcePrefix availableMemory
        | "SaltAndPepperNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let probability = parameterValue "probability"
            $"|> saltAndPepperNoise<{pixelType}> {width} {height} {depth} {probability}" |> sourcePrefix availableMemory
        | "ShotNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let scale = parameterValue "scale"
            $"|> shotNoise<{pixelType}> {width} {height} {depth} {scale}" |> sourcePrefix availableMemory
        | "SpeckleNoise" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let std = parameterValue "std"
            $"|> speckleNoise<{pixelType}> {width} {height} {depth} {std}" |> sourcePrefix availableMemory
        | "CreateByEuler2DTransform" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "UInt8" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let boxSize = parameterValue "boxSize"
            let transform = savedParamValue "transform" node
            let suffix = safeIdentifier node.Id
            let imageName = $"eulerImage_{suffix}"
            let transformName = $"eulerTransform_{suffix}"
            let transformBody =
                match transform.Trim().ToLowerInvariant() with
                | "antidiagonal"
                | "anti diagonal"
                | "anti-diagonal" ->
                    $"(offset, offset, a), (float width - dx - offset, dx - offset)"
                | "topdown"
                | "top down"
                | "top-down" ->
                    $"(offset, offset, a), (float width / 2.0 - offset, dx - offset)"
                | _ ->
                    $"(offset, offset, a), (0.0, 0.0)"

            String.concat Environment.NewLine
                [ $"let width = {width}"
                  $"let height = {height}"
                  $"let depth = {depth}"
                  $"let boxSize = int {boxSize}"
                  $"let {imageName} = Image<{pixelType}>([width; height])"
                  $"for i in [0..boxSize-1] do"
                  $"    for j in [0..boxSize-1] do"
                  $"        {imageName}[i,j] <- LanguagePrimitives.GenericOne"
                  $"let {transformName} (i: uint) : (float * float * float) * (float * float) ="
                  $"    let dx = float i"
                  $"    let a = 2.0 * System.Math.PI * float i / float depth"
                  $"    let offset = float boxSize / 2.0 - 0.5"
                  $"    {transformBody}"
                  $"debug 1u {uint64Literal availableMemory}"
                  $"|> createByEuler2DTransform<{pixelType}> {imageName} depth {transformName}" ]
        | "ReadRandom" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let depth = parameterValue "depth"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readRandom<{pixelType}> {depth} {input} {suffix}" |> sourcePrefix availableMemory
        | "EstimateHistogram" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let slices = parameterValue "slices"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            let down = parameterValue "down"
            let estimator = quotedParameter "estimator"
            let confidence = parameterValue "confidence"
            $"|> estimateHistogram<{pixelType}> {slices} {input} {suffix} {down} {estimator} {confidence}" |> sourcePrefix availableMemory
        | "ReadRange" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let first = quotedParameter "first"
            let step = parameterValue "step"
            let last = quotedParameter "last"
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readRange<{pixelType}> {first} {step} {last} {input} {suffix}" |> sourcePrefix availableMemory
        | "ReadSlab" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> readSlab<{pixelType}> {input} {suffix}" |> sourcePrefix availableMemory
        | "ReadZarrSlab" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "UInt8" node
            let input = quotedParameter "input"
            let slabDepth = parameterValue "slabDepth"
            let multiscaleIndex = parameterValue "multiscaleIndex"
            let datasetIndex = parameterValue "datasetIndex"
            let timepoint = parameterValue "timepoint"
            let channel = parameterValue "channel"
            let maxParallelChunks = parameterValue "maxParallelChunks"
            $"|> readZarrSlab<{pixelType}> {input} {slabDepth} {multiscaleIndex} {datasetIndex} {timepoint} {channel} {maxParallelChunks}" |> sourcePrefix availableMemory
        | "ReadNexusSlab" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "UInt16" node
            let input = quotedParameter "input"
            let datasetPath = quotedParameter "datasetPath"
            let slabDepth = parameterValue "slabDepth"
            let frameAxis = parameterValue "frameAxis"
            let yAxis = parameterValue "yAxis"
            let xAxis = parameterValue "xAxis"
            $"|> readNexusSlab<{pixelType}> {input} {datasetPath} {slabDepth} {frameAxis} {yAxis} {xAxis}" |> sourcePrefix availableMemory
        | "ReadPointSet" ->
            let availableMemory = parameterValue "availableMemory"
            let input = quotedParameter "input"
            $"|> readPointSet {input}" |> sourcePrefix availableMemory
        | "Read" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            $"|> read<{pixelType}> {input} {suffix}" |> sourcePrefix availableMemory
        | "ReadVolume" ->
            let availableMemory = parameterValue "availableMemory"
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let input = quotedParameter "input"
            $"|> readVolume<{pixelType}> {input}" |> sourcePrefix availableMemory
        | "Resize" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let width = parameterValue "width"
            let height = parameterValue "height"
            let depth = parameterValue "depth"
            let interpolation = quotedParameter "interpolation"
            $"|> resize<{pixelType}> {width} {height} {depth} {interpolation}"
        | "Resample" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let factorX = parameterValue "factorX"
            let factorY = parameterValue "factorY"
            let factorZ = parameterValue "factorZ"
            let interpolation = quotedParameter "interpolation"
            $"|> resample<{pixelType}> {factorX} {factorY} {factorZ} {interpolation}"
        | "WriteInSlabs" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> writeInSlabs {output} {suffix} {chunkX} {chunkY} {chunkZ}"
        | "WriteZarr" ->
            let output = quotedParameter "output"
            let name = quotedParameter "name"
            let depth = parameterValue "depth"
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            let physicalSizeX = parameterValue "physicalSizeX"
            let physicalSizeY = parameterValue "physicalSizeY"
            let physicalSizeZ = parameterValue "physicalSizeZ"
            let maxConcurrentWrites = parameterValue "maxConcurrentWrites"
            $">=> writeZarr {output} {name} {depth} {chunkX} {chunkY} {chunkZ} {physicalSizeX} {physicalSizeY} {physicalSizeZ} {maxConcurrentWrites}"
        | "WriteVolume" ->
            let output = quotedParameter "output"
            $">=> writeVolume {output}"
        | "WriteMesh" ->
            let output = quotedParameter "output"
            let format = quotedParameter "format"
            $">=> writeMesh {output} {format}"
        | "WritePointSet" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> writePointSet {output} {suffix}"
        | "WriteMatrix" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> writeMatrix {output} {suffix}"
        | "WriteCSV" ->
            let output = quotedParameter "output"
            match savedParamValue "dataKind" node with
            | "Matrix" -> $">=> writeCSVMatrix {output}"
            | "Histogram" -> $">=> writeCSVHistogram {output}"
            | _ -> $">=> writeCSVPointSet {output}"
        | "WriteNexus" ->
            let output = quotedParameter "output"
            let datasetPath = quotedParameter "datasetPath"
            let depth = parameterValue "depth"
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            let frameAxis = parameterValue "frameAxis"
            let yAxis = parameterValue "yAxis"
            let xAxis = parameterValue "xAxis"
            $">=> writeNexus {output} {datasetPath} {depth} {chunkX} {chunkY} {chunkZ} {frameAxis} {yAxis} {xAxis}"
        | "WriteThrough" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            $">=> write {output} {suffix}"
        | "WriteSlabSlices" ->
            let output = quotedParameter "output"
            let suffix = quotedParameter "suffix"
            let windowSize = parameterValue "windowSize"
            $">=> teeFst (writeSlabSlices {output} {suffix} {windowSize})"
        | "Tap" ->
            let tapPrintNode =
                graph.Edges
                |> Array.choose (fun edge ->
                    if edge.FromNode = node.Id
                       && edge.FromKind = "scalarOutput"
                       && edge.ToKind = "parameterInput" then
                        nodesById |> Map.tryFind edge.ToNode
                    else
                        None)
                |> Array.filter (fun candidate -> candidate.FunctionId = "Print")
                |> Array.distinctBy _.Id
                |> Array.tryHead

            match tapPrintNode with
            | Some printNode ->
                let tapInputName =
                    printNode.Parameters
                    |> Seq.mapi (fun index parameter -> index, parameter)
                    |> Seq.tryPick (fun (parameterIndex, parameter) ->
                        if parameter.Key.StartsWith("input", StringComparison.Ordinal)
                           && parameter.UseInput
                           && graph.Edges
                              |> Array.exists (fun edge ->
                                  edge.ToNode = printNode.Id
                                  && edge.ToKind = "parameterInput"
                                  && edge.ToPort = parameterIndex
                                  && edge.FromNode = node.Id
                                  && edge.FromKind = "scalarOutput") then
                            Some(printInputNameForNode printNode parameter.Key parameterIndex)
                        else
                            None)
                    |> Option.defaultValue "I"

                let tapVariable = safeIdentifier tapInputName

                let inputs =
                    printNode.Parameters
                    |> Seq.mapi (fun index parameter -> index, parameter)
                    |> Seq.choose (fun (parameterIndex, parameter) ->
                        if parameter.Key.StartsWith("input", StringComparison.Ordinal) && parameter.UseInput then
                            let expression =
                                graph.Edges
                                |> Array.tryFind (fun edge ->
                                    edge.ToNode = printNode.Id
                                    && edge.ToKind = "parameterInput"
                                    && edge.ToPort = parameterIndex)
                                |> Option.bind (fun edge ->
                                    if edge.FromNode = node.Id && edge.FromKind = "scalarOutput" then
                                        Some tapVariable
                                    else
                                        Some(parameterValueForNode printNode parameter.Key))
                                |> Option.defaultValue (parameterValueForNode printNode parameter.Key)

                            Some(parameter.Key, printInputNameForNode printNode parameter.Key parameterIndex, expression)
                        else
                            None)
                    |> Seq.toList

                let format = interpolatedStringExpression (savedParamValue "format" printNode) inputs
                $">=> tapIt (fun {tapVariable} -> {format})"
            | None ->
                let label = stringParameter "label"
                $">=> tap {label}"
        | "Print" ->
            let inputs =
                node.Parameters
                |> Seq.mapi (fun index parameter -> index, parameter)
                |> Seq.choose (fun (parameterIndex, parameter) ->
                    if parameter.Key.StartsWith("input", StringComparison.Ordinal) && parameter.UseInput then
                        Some(parameter.Key, printInputName parameter.Key parameterIndex, parameterValue parameter.Key)
                    else
                        None)
                |> Seq.toList


            let format = interpolatedStringExpression (savedParamValue "format" node) inputs

            $"printfn {format}"
        | "Histogram" ->
            ">=> histogram () >=> map2pairs --> pairs2floats --> plot (showChartXY \"Column\")"
        | "HistogramData" ->
            ">=> histogram ()"
        | "Chart" ->
            let values = parameterValue "input"
            let kind = savedParamValue "kind" node
            $"showChart \"{kind}\" {values}"
        | "ShowImage" ->
            ">=> show showImagePlot"
        | "SumProjection" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let functionName = savedParamValue "function" node
            $">=> sumProjection<{pixelType}> {quote functionName}"
        | "UnaryImageFunction" ->
            $">=> {unaryImageFunctionName node}"
        | "ComplexRe" ->
            ">=> Re"
        | "ComplexIm" ->
            ">=> Im"
        | "ComplexModulus" ->
            ">=> modulus"
        | "ComplexArg" ->
            ">=> arg"
        | "ComplexConjugate" ->
            ">=> conjugate"
        | "FFT" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> FFT<{pixelType}> {chunkX} {chunkY} {chunkZ}"
        | "InvFFT" ->
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> invFFT {chunkX} {chunkY} {chunkZ}"
        | "ShiftFFT" ->
            let chunkX = parameterValue "chunkX"
            let chunkY = parameterValue "chunkY"
            let chunkZ = parameterValue "chunkZ"
            $">=> shiftFFT {chunkX} {chunkY} {chunkZ}"
        | "VectorElement" ->
            let componentId = parameterValue "component"
            $">=> vectorElement<float> {componentId}"
        | "VectorMapElements" ->
            let functionName = quotedParameter "function"
            $">=> vectorMapElements {functionName}"
        | "VectorAngleTo" ->
            let x = parameterValue "x"
            let y = parameterValue "y"
            let z = parameterValue "z"
            $">=> vectorAngleTo [ {x}; {y}; {z} ]"
        | "Gradient" ->
            let order = parameterValue "order"
            let windowSize = parameterValue "windowSize"
            $">=> gradient {order} (Some {windowSize})"
        | "StructureTensor" ->
            let sigma = parameterValue "sigma"
            let rho = parameterValue "rho"
            $">=> structureTensor {sigma} {rho}"
        | "PCA" ->
            let components = parameterValue "components"
            $">=> PCA {components}"
        | id when isScalarImageFunction id ->
            let value = parameterValue "value"
            $">=> {scalarImageFunctionName node |> Option.get} {value}"
        | id when pairStageFunctionName node |> Option.isSome ->
            $"{pairCompositionOperator node} {pairStageFunctionName node |> Option.get}"
        | "SmoothWGauss" ->
            let sigma = parameterValue "sigma"
            let outputRegionMode = parameterValue "outputRegionMode" |> optionQualified "ImageFunctions"
            let boundaryCondition = parameterValue "boundaryCondition" |> optionQualified "ImageFunctions"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> smoothWGauss {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | "Convolve" ->
            let kernel = parameterValue "kernel"
            let outputRegionMode = parameterValue "outputRegionMode" |> optionQualified "ImageFunctions"
            let boundaryCondition = parameterValue "boundaryCondition" |> optionQualified "ImageFunctions"
            let windowSize = parameterValue "windowSize" |> optionUInt
            $">=> convolve {kernel} {outputRegionMode} {boundaryCondition} {windowSize}"
        | "FiniteDiff" ->
            let axis1 = parameterValue "axis1"
            let axis2 = parameterValue "axis2"
            $">=> finiteDiff {axis1} {axis2}"
        | "Clamp" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let lower = parameterValue "lower"
            let upper = parameterValue "upper"
            $">=> clamp<{pixelType}> {lower} {upper}"
        | "ShiftScale" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let shift = parameterValue "shift"
            let scale = parameterValue "scale"
            $">=> shiftScale<{pixelType}> {shift} {scale}"
        | "IntensityStretch" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let inputMinimum = parameterValue "inputMinimum"
            let inputMaximum = parameterValue "inputMaximum"
            let outputMinimum = parameterValue "outputMinimum"
            let outputMaximum = parameterValue "outputMaximum"
            $">=> intensityStretch<{pixelType}> {inputMinimum} {inputMaximum} {outputMinimum} {outputMaximum}"
        | "HistogramEqualization" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let histogram = parameterValue "histogram"
            $">=> histogramEqualization<{pixelType}> {histogram}"
        | "CreatePadding" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let beforeX = parameterValue "beforeX"
            let afterX = parameterValue "afterX"
            let beforeY = parameterValue "beforeY"
            let afterY = parameterValue "afterY"
            let beforeZ = parameterValue "beforeZ"
            let afterZ = parameterValue "afterZ"
            let value = parameterValue "value"
            $">=> createPadding<{pixelType}> {beforeX} {afterX} {beforeY} {afterY} {beforeZ} {afterZ} {value}"
        | "Crop" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let beforeX = parameterValue "beforeX"
            let afterX = parameterValue "afterX"
            let beforeY = parameterValue "beforeY"
            let afterY = parameterValue "afterY"
            let beforeZ = parameterValue "beforeZ"
            let afterZ = parameterValue "afterZ"
            $">=> crop<{pixelType}> {beforeX} {afterX} {beforeY} {afterY} {beforeZ} {afterZ}"
        | "SmoothWMedian" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> smoothWMedian<{pixelType}> {radius} {windowSize}"
        | "SmoothWBilateral" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let domainSigma = parameterValue "domainSigma"
            let rangeSigma = parameterValue "rangeSigma"
            let windowSize = parameterValue "windowSize"
            $">=> smoothWBilateral<{pixelType}> {domainSigma} {rangeSigma} {windowSize}"
        | "GradientMagnitude" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let windowSize = parameterValue "windowSize"
            $">=> gradientMagnitude<{pixelType}> {windowSize}"
        | "SobelEdge" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let windowSize = parameterValue "windowSize"
            $">=> sobelEdge<{pixelType}> {windowSize}"
        | "Laplacian" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let windowSize = parameterValue "windowSize"
            $">=> laplacian<{pixelType}> {windowSize}"
        | "GrayscaleErode" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> grayscaleErode<{pixelType}> {radius} {windowSize}"
        | "GrayscaleDilate" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> grayscaleDilate<{pixelType}> {radius} {windowSize}"
        | "GrayscaleOpening" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> grayscaleOpening<{pixelType}> {radius} {windowSize}"
        | "GrayscaleClosing" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> grayscaleClosing<{pixelType}> {radius} {windowSize}"
        | "WhiteTopHat" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> whiteTopHat<{pixelType}> {radius} {windowSize}"
        | "BlackTopHat" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> blackTopHat<{pixelType}> {radius} {windowSize}"
        | "MorphologicalGradient" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> morphologicalGradient<{pixelType}> {radius} {windowSize}"
        | "ImageComparison" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            $">=> {comparisonStageFunctionName node}<{pixelType}>"
        | "MaskLogic" ->
            $">=> {maskLogicStageFunctionName node}"
        | "MaskNot" ->
            ">=> maskNot"
        | "BinaryContour" ->
            let fullyConnected = parameterValue "fullyConnected"
            let windowSize = parameterValue "windowSize"
            $">=> binaryContour {fullyConnected} {windowSize}"
        | "BinaryMedian" ->
            let radius = parameterValue "radius"
            let windowSize = parameterValue "windowSize"
            $">=> binaryMedian {radius} {windowSize}"
        | "RemoveSmallObjects" ->
            let maximumVolume = parameterValue "maximumVolume"
            let connectivity = parameterValue "connectivity"
            $">=> removeSmallObjects {maximumVolume} ObjectConnectivity.{connectivity}"
        | "FillSmallHoles" ->
            let maximumVolume = parameterValue "maximumVolume"
            let connectivity = parameterValue "connectivity"
            $">=> fillSmallHoles {maximumVolume} ObjectConnectivity.{connectivity}"
        | "LabelContour" ->
            let pixelType = pixelTypeNameFromParameter "type" "UInt64" node
            let fullyConnected = parameterValue "fullyConnected"
            let windowSize = parameterValue "windowSize"
            $">=> labelContour<{pixelType}> {fullyConnected} {windowSize}"
        | "ChangeLabel" ->
            let pixelType = pixelTypeNameFromParameter "type" "UInt64" node
            let fromLabel = parameterValue "fromLabel"
            let toLabel = parameterValue "toLabel"
            $">=> changeLabel<{pixelType}> {fromLabel} {toLabel}"
        | "ComputeStats" ->
            ">=> computeStats ()"
        | "SurfaceArea" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> surfaceArea {xUnit} {yUnit} {zUnit}"
        | "Volume" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> volume {xUnit} {yUnit} {zUnit}"
        | "FitBiasModel" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let order = parameterValue "order"
            let depth = parameterValue "depth"
            $">=> fitBiasModel<{pixelType}> {order} {depth}"
        | "FitBiasModelMasked" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let order = parameterValue "order"
            let depth = parameterValue "depth"
            $">=> fitBiasModelMasked<{pixelType}> {order} {depth}"
        | "CorrectBias" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let model = parameterValue "model"
            $">=> correctBias<{pixelType}> {model}"
        | "CorrectBiasMasked" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let model = parameterValue "model"
            $">=> correctBiasMasked<{pixelType}> {model}"
        | "SerialPolynomialBiasCorrect" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let order = parameterValue "order"
            $">=> serialPolynomialBiasCorrect<{pixelType}> {order}"
        | "SerialEstTrans" ->
            let pixelType = pipelinePixelType "Float64"
            let maxShift = parameterValue "maxShift"
            let method = quotedParameter "method"
            let sigma0 = parameterValue "sigma0"
            let scaleFactor = parameterValue "scaleFactor"
            let scaleLevels = parameterValue "scaleLevels"
            let contrastThreshold = parameterValue "contrastThreshold"
            let maxKeypoints = parameterValue "maxKeypoints"
            let matchTolerance = parameterValue "matchTolerance"
            let maxIterations = parameterValue "maxIterations"
            let initialLinearStep = parameterValue "initialLinearStep"
            let initialTranslationStep = parameterValue "initialTranslationStep"
            let minStep = parameterValue "minStep"
            let stepShrink = parameterValue "stepShrink"
            $">=> serialEstTrans<{pixelType}> {maxShift} {method} {sigma0} {scaleFactor} {scaleLevels} {contrastThreshold} {maxKeypoints} {matchTolerance} {maxIterations} {initialLinearStep} {initialTranslationStep} {minStep} {stepShrink}"
        | "SerialApplyTrans" ->
            let pixelType = pipelinePixelType "Float64"
            let background = parameterValue "background"
            $">=> serialApplyTrans<{pixelType}> {background}"
        | "SerialApplyManifestInBoundingBox" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let manifest = parameterValue "manifest"
            let background = parameterValue "background"
            $">=> serialApplyManifestInBoundingBox<{pixelType}> {manifest} {background}"
        | "PointPairDistances" ->
            let xUnit = parameterValue "xUnit"
            let yUnit = parameterValue "yUnit"
            let zUnit = parameterValue "zUnit"
            $">=> pointPairDistances {xUnit} {yUnit} {zUnit}"
        | "AddNormalNoise" ->
            let mean = parameterValue "mean"
            let std = parameterValue "std"
            $">=> addNormalNoise {mean} {std}"
        | "AddSaltAndPepperNoise" ->
            let probability = parameterValue "probability"
            $">=> addSaltAndPepperNoise {probability}"
        | "AddShotNoise" ->
            let scale = parameterValue "scale"
            $">=> addShotNoise {scale}"
        | "AddSpeckleNoise" ->
            let std = parameterValue "std"
            $">=> addSpeckleNoise {std}"
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
        | "RelabelComponents" ->
            let minimumObjectSize = parameterValue "minimumObjectSize"
            let windowSize = parameterValue "windowSize"
            $">=> relabelComponents {minimumObjectSize} {windowSize}"
        | "MarchingCubes" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let surfaceValue = parameterValue "surfaceValue"
            $">=> marchingCubes<{pixelType}> {surfaceValue}"
        | "DogKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma0 = parameterValue "sigma0"
            let scaleFactor = parameterValue "scaleFactor"
            let scaleLevels = parameterValue "scaleLevels"
            let contrastThreshold = parameterValue "contrastThreshold"
            let stride = parameterValue "stride"
            $">=> dogKeypoints<{pixelType}> {sigma0} {scaleFactor} {scaleLevels} {contrastThreshold} {stride}"
        | "SiftKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma0 = parameterValue "sigma0"
            let scaleFactor = parameterValue "scaleFactor"
            let scaleLevels = parameterValue "scaleLevels"
            let contrastThreshold = parameterValue "contrastThreshold"
            let stride = parameterValue "stride"
            $">=> siftKeypoints<{pixelType}> {sigma0} {scaleFactor} {scaleLevels} {contrastThreshold} {stride}"
        | "LogBlobKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma = parameterValue "sigma"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> logBlobKeypoints<{pixelType}> {sigma} {threshold} {stride}"
        | "HessianKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma = parameterValue "sigma"
            let responseKind = quotedParameter "responseKind"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> hessianKeypoints<{pixelType}> {sigma} {responseKind} {threshold} {stride}"
        | "Harris3DKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma = parameterValue "sigma"
            let rho = parameterValue "rho"
            let k = parameterValue "k"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> harris3DKeypoints<{pixelType}> {sigma} {rho} {k} {threshold} {stride}"
        | "Forstner3DKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma = parameterValue "sigma"
            let rho = parameterValue "rho"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> forstner3DKeypoints<{pixelType}> {sigma} {rho} {threshold} {stride}"
        | "PhaseCongruencyKeypoints" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float64" node
            let sigma = parameterValue "sigma"
            let threshold = parameterValue "threshold"
            let stride = parameterValue "stride"
            $">=> phaseCongruencyKeypoints<{pixelType}> {sigma} {threshold} {stride}"
        | "StreamConnectedObjects" ->
            let connectivity = parameterValue "connectivity"
            $">=> streamConnectedObjects<uint8> ObjectConnectivity.{connectivity}"
        | "PaintObjects" ->
            let width = parameterValue "width"
            let height = parameterValue "height"
            $">=> paintObjects {width} {height}"
        | "PaintObjectsCropped" ->
            $">=> paintObjectsCropped"
        | "SignedDistanceBand" ->
            let bandRadius = parameterValue "bandRadius"
            let stride = parameterValue "stride"
            $">=> signedDistanceBand {bandRadius} {stride}"
        | "ComponentTranslationTable" ->
            let windowSize = parameterValue "windowSize"
            $">=> makeConnectedComponentTranslationTable {windowSize}"
        | "CollapseComponentLabels" ->
            let windowSize = parameterValue "windowSize"
            let translationTable = parameterValue "translationTable"
            $">=> updateConnectedComponents {windowSize} {translationTable}"
        | "PermuteAxes" ->
            let axes = parameterValue "axes"
            let tileSize = parameterValue "tileSize"
            $">=> permuteAxes {axes} {tileSize}"
        | "ResampleAffineTrilinearSlices" ->
            let pixelType = pixelTypeNameFromParameter "type" "Float32" node
            let input = quotedParameter "input"
            let suffix = quotedParameter "suffix"
            let lerp = parameterValue "lerp"
            let windowSize = parameterValue "windowSize"
            let inputGeometry = parameterValue "inputGeometry"
            let outputGeometry = parameterValue "outputGeometry"
            let affine = parameterValue "affine"
            let background = parameterValue "background"
            $"resampleAffineTrilinearSlices<{pixelType}> {input} {suffix} {lerp} {windowSize} {inputGeometry} {outputGeometry} {affine} {background} |> Seq.iter ignore"
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
        let graph = replaceFileDirectorySelectors graph
        let builder = StringBuilder()
        let nodesById = graph.Nodes |> Seq.map (fun node -> node.Id, node) |> Map.ofSeq
        let scalarNodes =
            graph.Nodes
            |> Array.filter (fun node ->
                node.FunctionId = "Scalar"
                || node.FunctionId = "ScalarOp"
                || node.FunctionId = "ScalarFunction"
                || node.FunctionId = "OtsuThresholdFromHistogram"
                || node.FunctionId = "MomentsThresholdFromHistogram")

        let scalarNamesByNodeId = scalarNames scalarNodes
        let statsNodesWithLinkedFields =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter (fun node -> node.FunctionId = "ComputeStats" || isSingleValueReducerNode node)
            |> Array.distinctBy _.Id

        let statsNamesByNodeId =
            statsNodesWithLinkedFields
            |> Array.mapi (fun index node ->
                let name =
                    if node.FunctionId = "ComputeStats" then $"ImageStats{index}"
                    else $"{node.FunctionId}{index}"
                node.Id, name)
            |> Map.ofArray

        let translationTableNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isTranslationTableNode
            |> Array.distinctBy _.Id

        let translationTableNamesByNodeId =
            translationTableNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"TranslationTable{index}")
            |> Map.ofArray

        let histogramNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isHistogramDataNode
            |> Array.distinctBy _.Id

        let histogramNamesByNodeId =
            histogramNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"Histogram{index}")
            |> Map.ofArray

        let quantileNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isQuantilesNode
            |> Array.distinctBy _.Id

        let quantileNamesByNodeId =
            quantileNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"Quantiles{index}")
            |> Map.ofArray

        let stackInfoNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isStackInfoNode
            |> Array.distinctBy _.Id

        let stackInfoNamesByNodeId =
            stackInfoNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"StackInfo{index}")
            |> Map.ofArray

        let chunkInfoNodesWithLinkedOutputs =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "reducerOutput" && edge.ToKind = "parameterInput" then
                    nodesById |> Map.tryFind edge.FromNode
                else
                    None)
            |> Array.filter isChunkInfoNode
            |> Array.distinctBy _.Id

        let chunkInfoNamesByNodeId =
            chunkInfoNodesWithLinkedOutputs
            |> Array.mapi (fun index node -> node.Id, $"ChunkInfo{index}")
            |> Map.ofArray

        let newLine = Environment.NewLine

        let dataEdges =
            graph.Edges
            |> Array.filter (fun edge -> edge.FromKind <> "scalarOutput" && edge.ToKind <> "parameterInput")

        let printNodesUsedByTap =
            graph.Edges
            |> Array.choose (fun edge ->
                if edge.FromKind = "scalarOutput" && edge.ToKind = "parameterInput" then
                    match nodesById |> Map.tryFind edge.FromNode, nodesById |> Map.tryFind edge.ToNode with
                    | Some sourceNode, Some targetNode when sourceNode.FunctionId = "Tap" && targetNode.FunctionId = "Print" ->
                        Some targetNode.Id
                    | _ ->
                        None
                else
                    None)
            |> Set.ofArray

        let bindingNameForOutput edge =
            match edge.FromKind with
            | "scalarOutput" ->
                scalarNamesByNodeId |> Map.tryFind edge.FromNode
            | "reducerOutput" ->
                match statsNamesByNodeId |> Map.tryFind edge.FromNode with
                | Some name -> Some name
                | None ->
                    match translationTableNamesByNodeId |> Map.tryFind edge.FromNode with
                    | Some name -> Some name
                    | None ->
                        match histogramNamesByNodeId |> Map.tryFind edge.FromNode with
                        | Some name -> Some name
                        | None ->
                            match quantileNamesByNodeId |> Map.tryFind edge.FromNode with
                            | Some name -> Some name
                            | None ->
                                match stackInfoNamesByNodeId |> Map.tryFind edge.FromNode with
                                | Some name -> Some name
                                | None -> chunkInfoNamesByNodeId |> Map.tryFind edge.FromNode
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

        let commonUpstreamInputEdge (node: SavedNode) =
            match incomingDataEdge node.Id 0 with
            | Some edge -> Some edge
            | None ->
                Some
                    { FromNode = node.Id
                      FromKind = "dataOutput"
                      FromPort = 0
                      ToNode = node.Id
                      ToKind = "dataInput"
                      ToPort = 0 }

        let stageCall (node: SavedNode) =
            let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node

            if line.StartsWith(">=> ", StringComparison.Ordinal) then
                Some(line.Substring(4))
            elif line.StartsWith(">>=> ", StringComparison.Ordinal) then
                Some(line.Substring(5))
            else
                None

        let rec inferredImagePixelType visited (targetNode: SavedNode) inputPort =
            if visited |> Set.contains targetNode.Id then
                None
            else
                let visited = visited |> Set.add targetNode.Id

                incomingDataEdge targetNode.Id inputPort
                |> Option.bind (fun edge ->
                    nodesById
                    |> Map.tryFind edge.FromNode
                    |> Option.bind (fun sourceNode ->
                        match sourceNode.FunctionId, edge.FromPort with
                        | "SerialEstTrans", (0 | 1) ->
                            inferredImagePixelType visited sourceNode 0
                        | "Cast", 0 ->
                            Some(pixelTypeNameFromParameter "targetType" "Float64" sourceNode)
                        | _ ->
                            let configured = savedParamValue "type" sourceNode
                            if String.IsNullOrWhiteSpace configured then None else Some(pixelTypeNameFromSuffix configured)))

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
                        | Some upstream ->
                            let upstreamExpression = pipelineExpression visited upstream
                            if upstream.FunctionId = "StructureTensor" && edge.FromPort >= 0 && edge.FromPort <= 3 then
                                $"{upstreamExpression}{newLine}>=> selectGroupedOutput 4u {edge.FromPort}u"
                            elif upstream.FunctionId = "PCA" && edge.FromPort >= 0 && edge.FromPort <= 8 then
                                let rawComponents = savedParamValue "components" upstream
                                let components =
                                    if String.IsNullOrWhiteSpace rawComponents then "3u"
                                    else numericLiteral UInt32 rawComponents
                                $"{upstreamExpression}{newLine}>=> selectGroupedOutput ({components} + 1u) {edge.FromPort}u"
                            elif upstream.FunctionId = "AffineRegistration" && edge.FromPort >= 0 && edge.FromPort <= 1 then
                                $"{upstreamExpression}{newLine}>=> selectGroupedValueOutput 2u {edge.FromPort}u"
                            elif upstream.FunctionId = "SerialEstTrans" && edge.FromPort = 0 then
                                let pixelType =
                                    inferredImagePixelType Set.empty upstream 0
                                    |> Option.defaultValue (pixelTypeNameFromParameter "type" "Float64" upstream)
                                $"{upstreamExpression}{newLine}>=> serialTransImage<{pixelType}>"
                            elif upstream.FunctionId = "SerialEstTrans" && edge.FromPort = 1 then
                                let pixelType =
                                    inferredImagePixelType Set.empty upstream 0
                                    |> Option.defaultValue (pixelTypeNameFromParameter "type" "Float64" upstream)
                                $"{upstreamExpression}{newLine}>=> serialTransManifest<{pixelType}>"
                            elif upstream.FunctionId = "EstimateHistogram" && edge.FromPort = 0 then
                                $"{upstreamExpression}{newLine}>=> histogramEstimateMap"
                            else
                                upstreamExpression
                        | None -> $"// Cannot generate F#: missing upstream node {edge.FromNode}"
                    | None -> $"// Cannot generate F#: missing input {port} for {node.FunctionId}"

                let rec branchStageExpression commonInputEdge visited (node: SavedNode) =
                    if visited |> Set.contains node.Id then
                        None
                    elif node.Id = commonInputEdge.FromNode then
                        Some "identity"
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
                            let fanOut =
                                formatStageTuple
                                    "identity"
                                    "identity"
                            pairStageFunctionName node
                            |> Option.map (fun pairFunction -> $"{shared}{newLine}{fanOut}{newLine}{pairCompositionOperator node} {pairFunction}"))
                    | Some leftEdge, Some rightEdge ->
                        match nodesById |> Map.tryFind leftEdge.FromNode, nodesById |> Map.tryFind rightEdge.FromNode with
                        | Some leftNode, Some rightNode ->
                            match commonUpstreamInputEdge leftNode, commonUpstreamInputEdge rightNode with
                            | Some leftInput, Some rightInput when sameOutputEdge leftInput rightInput ->
                                match nodesById |> Map.tryFind leftInput.FromNode with
                                | Some sharedNode ->
                                    match branchStageExpression leftInput Set.empty leftNode, branchStageExpression leftInput Set.empty rightNode with
                                    | Some leftStage, Some rightStage ->
                                        let shared = pipelineExpression visited sharedNode
                                        pairStageFunctionName node
                                        |> Option.map (fun pairFunction -> $"{shared}{newLine}{formatStageTuple leftStage rightStage}{newLine}{pairCompositionOperator node} {pairFunction}")
                                    | _ -> None
                                | None -> None
                            | _ -> None
                        | _ -> None
                    | _ -> None

                match node.FunctionId with
                | "SerialApplyTrans" ->
                    match incomingDataEdge node.Id 0, incomingDataEdge node.Id 1, stageCall node with
                    | Some imageEdge, Some manifestEdge, Some stage
                        when imageEdge.FromNode = manifestEdge.FromNode
                             && imageEdge.FromKind = manifestEdge.FromKind
                             && imageEdge.FromPort = 0
                             && manifestEdge.FromPort = 1 ->
                        match nodesById |> Map.tryFind imageEdge.FromNode with
                        | Some upstream when upstream.FunctionId = "SerialEstTrans" ->
                            let input = pipelineExpression visited upstream
                            $"{input}{newLine}>=> {stage}"
                        | _ ->
                            let left = inputExpression 0 |> parenthesizeBlock
                            let right = inputExpression 1 |> parenthesizeBlock
                            $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}>=> {stage}"
                    | _ ->
                        let stage = stageCall node |> Option.get
                        let left = inputExpression 0 |> parenthesizeBlock
                        let right = inputExpression 1 |> parenthesizeBlock
                        $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}>=> {stage}"
                | id when pairStageFunctionName node |> Option.isSome ->
                    match sharedFanOutExpression () with
                    | Some expression ->
                        expression
                    | None ->
                        let pairFunction = pairStageFunctionName node |> Option.get
                        let left = inputExpression 0
                        let right = inputExpression 1
                        let left = parenthesizeBlock left
                        let right = parenthesizeBlock right
                        $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}{pairCompositionOperator node} {pairFunction}"
                | _ when (incomingDataEdge node.Id 1 |> Option.isSome) && (stageCall node |> Option.isSome) ->
                    let stage = stageCall node |> Option.get
                    let left = inputExpression 0 |> parenthesizeBlock
                    let right = inputExpression 1 |> parenthesizeBlock
                    $"({newLine}{indentBlock 4 left},{newLine}{indentBlock 4 right}{newLine}){newLine}||> zip{newLine}>=> {stage}"
                | _ ->
                    let line = savedElementLine graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node

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
                | "WriteThrough"
                | "WriteVolume"
                | "WriteInSlabs"
                | "WriteZarr"
                | "WriteMesh"
                | "WritePointSet"
                | "WriteMatrix"
                | "WriteCSV"
                | "WriteNexus"
                | "Histogram"
                | "ShowImage" ->
                    $"{expression}{newLine}|> sink"
                | "ComputeStats" ->
                    $"{expression}{newLine}|> drain"
                | "SurfaceArea"
                | "Volume"
                | "PointPairDistances"
                | "FitBiasModel"
                | "FitBiasModelMasked"
                | "AffineRegistration" ->
                    $"{expression}{newLine}|> drain"
                | "ComponentTranslationTable" ->
                    $"{expression}{newLine}|> drain"
                | "HistogramData" ->
                    $"{expression}{newLine}|> drain"
                | "EstimateHistogram" ->
                    $"{expression}{newLine}|> drain"
                | _ ->
                    expression

            let terminalNodes =
                graph.Nodes
                |> Array.filter (fun node ->
                    node.FunctionId <> "Scalar"
                    && node.FunctionId <> "ScalarOp"
                    && node.FunctionId <> "ScalarFunction"
                    && node.FunctionId <> "OtsuThresholdFromHistogram"
                    && node.FunctionId <> "MomentsThresholdFromHistogram"
                    && node.FunctionId <> "GetStackInfo"
                    && node.FunctionId <> "GetChunkInfo"
                    && node.FunctionId <> "GetZarrInfo"
                    && node.FunctionId <> "GetNexusInfo"
                    && not (printNodesUsedByTap |> Set.contains node.Id)
                    && not (statsNamesByNodeId |> Map.containsKey node.Id)
                    && not (translationTableNamesByNodeId |> Map.containsKey node.Id)
                    && not (histogramNamesByNodeId |> Map.containsKey node.Id)
                    && not (quantileNamesByNodeId |> Map.containsKey node.Id)
                    && not (stackInfoNamesByNodeId |> Map.containsKey node.Id)
                    && not (chunkInfoNamesByNodeId |> Map.containsKey node.Id)
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
                    let name, text = scalarBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = text })

            let translationTableBindings =
                translationTableNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = translationTableNamesByNodeId |> Map.find node.Id
                    let expression = pipelineExpression Set.empty node
                    let body = indentBlock 4 $"{expression}{newLine}|> drain"

                    { Name = name
                      Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                      Text = $"let {name} ={newLine}{body}" })

            let histogramBindings =
                histogramNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = histogramNamesByNodeId |> Map.find node.Id
                    let expression = pipelineExpression Set.empty node
                    let body = indentBlock 4 $"{expression}{newLine}|> drain"

                    { Name = name
                      Dependencies = pipelineBindingDependencies Set.empty node |> Set.remove name
                      Text = $"let {name} ={newLine}{body}" })

            let quantileBindings =
                quantileNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name = quantileNamesByNodeId |> Map.find node.Id

                    let parameter key =
                        node.Parameters
                        |> Seq.tryFindIndex (fun parameter -> parameter.Key = key)
                        |> Option.map (fun index -> parameterExpression graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node index key)
                        |> Option.defaultValue { Value = savedParamValue key node; IsLinked = false }

                    let numericParameter key =
                        let expression = parameter key
                        if expression.IsLinked then expression.Value else numericLiteral Float64 expression.Value

                    let boolParameter key =
                        savedParamValue key node
                        |> fun value -> value.Trim().Equals("true", StringComparison.OrdinalIgnoreCase)

                    let quantileValues =
                        [ "q1", true
                          "q2", boolParameter "useQ2"
                          "q3", boolParameter "useQ3"
                          "q4", boolParameter "useQ4"
                          "q5", boolParameter "useQ5" ]
                        |> List.choose (fun (key, enabled) -> if enabled then Some(numericParameter key) else None)
                        |> String.concat "; "

                    let histogram = (parameter "histogram").Value

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = $"let {name} = quantiles [{quantileValues}] {histogram}" })

            let stackInfoBindings =
                stackInfoNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name, text = stackInfoBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = text })

            let chunkInfoBindings =
                chunkInfoNodesWithLinkedOutputs
                |> Array.map (fun node ->
                    let name, text = chunkInfoBinding graph nodesById scalarNamesByNodeId statsNamesByNodeId translationTableNamesByNodeId histogramNamesByNodeId quantileNamesByNodeId stackInfoNamesByNodeId chunkInfoNamesByNodeId node

                    { Name = name
                      Dependencies = parameterBindingDependencies node |> Set.remove name
                      Text = text })

            let bindings = Array.concat [ scalarBindings; statsBindings; translationTableBindings; histogramBindings; quantileBindings; stackInfoBindings; chunkInfoBindings ]
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

        let hasChart =
            graph.Nodes |> Array.exists (fun node -> node.FunctionId = "Histogram" || node.FunctionId = "Chart")

        let hasShowImage =
            graph.Nodes |> Array.exists (fun node -> node.FunctionId = "ShowImage")

        let hasVisualization = hasChart || hasShowImage

        builder.AppendLine("open StackProcessing") |> ignore

        if hasVisualization then
            builder.AppendLine("open Plotly.NET") |> ignore

        builder.AppendLine() |> ignore

        if hasChart then
            builder.AppendLine("let showChartData kind x y =") |> ignore
            builder.AppendLine("    match kind with") |> ignore
            builder.AppendLine("    | \"Scatter\" -> Chart.Scatter(x = x, y = y, mode = StyleParam.Mode.Markers)") |> ignore
            builder.AppendLine("    | \"Line\" -> Chart.Line(x = x, y = y)") |> ignore
            builder.AppendLine("    | \"Bar\" -> Chart.Bar(values = y, Keys = x)") |> ignore
            builder.AppendLine("    | \"Area\" -> Chart.Area(x = x, y = y)") |> ignore
            builder.AppendLine("    | \"Pie\" -> Chart.Pie(values = y, Labels = x)") |> ignore
            builder.AppendLine("    | \"Doughnut\" -> Chart.Doughnut(values = y, Labels = x)") |> ignore
            builder.AppendLine("    | _ -> Chart.Column(values = y, Keys = x)") |> ignore
            builder.AppendLine("    |> Chart.show") |> ignore
            builder.AppendLine() |> ignore
            builder.AppendLine("let showChart kind points =") |> ignore
            builder.AppendLine("    let x, y = points |> Map.toList |> List.unzip") |> ignore
            builder.AppendLine("    showChartData kind x y") |> ignore
            builder.AppendLine() |> ignore
            builder.AppendLine("let showChartXY kind x y =") |> ignore
            builder.AppendLine("    showChartData kind x y") |> ignore
            builder.AppendLine() |> ignore

        if hasShowImage then
            builder.AppendLine("let showImagePlot image =") |> ignore
            builder.AppendLine("    ImageFunctions.toSeqSeq image") |> ignore
            builder.AppendLine("    |> Chart.Heatmap") |> ignore
            builder.AppendLine("    |> Chart.show") |> ignore
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
