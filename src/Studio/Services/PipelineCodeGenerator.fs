namespace Studio.Services

open System
open System.Text
open Studio.Models

module PipelineCodeGenerator =

    let private paramValue key (state: PipelineNodeState) =
        state.Parameters
        |> Seq.tryFind (fun p -> p.Key = key)
        |> Option.map _.Value
        |> Option.defaultValue ""

    let private quote (value: string) =
        "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

    let private optionUInt (value: string) =
        let trimmed = value.Trim()
        if String.IsNullOrWhiteSpace trimmed || String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed.TrimEnd('u', 'U')}u)"

    let private optionRaw (value: string) =
        let trimmed = value.Trim()
        if String.IsNullOrWhiteSpace trimmed || String.Equals(trimmed, "None", StringComparison.OrdinalIgnoreCase) then
            "None"
        elif trimmed.StartsWith("Some", StringComparison.Ordinal) then
            trimmed
        else
            $"(Some {trimmed})"

    let private elementLine (state: PipelineNodeState) =
        match state.Kind with
        | PipelineElementKind.Source ->
            let availableMemory = paramValue "availableMemory" state
            $"source {availableMemory}"
        | PipelineElementKind.Read ->
            let pixelType = paramValue "pixelType" state
            let input = paramValue "input" state |> quote
            let suffix = paramValue "suffix" state |> quote
            $"|> read<{pixelType}> {input} {suffix}"
        | PipelineElementKind.DiscreteGaussian ->
            let sigma = paramValue "sigma" state
            let outputRegionMode = paramValue "outputRegionMode" state |> optionRaw
            let boundaryCondition = paramValue "boundaryCondition" state |> optionRaw
            let windowSize = paramValue "windowSize" state |> optionUInt
            $">=> discreteGaussian {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | PipelineElementKind.Cast ->
            let sourceType = paramValue "sourceType" state
            let targetType = paramValue "targetType" state
            $">=> cast<{sourceType},{targetType}>"
        | PipelineElementKind.Write ->
            let output = paramValue "output" state |> quote
            let suffix = paramValue "suffix" state |> quote
            $">=> write {output} {suffix}"
        | PipelineElementKind.Sink ->
            "|> sink"
        | _ ->
            $"// Unsupported element: {state.Title}"

    let generate (states: PipelineNodeState seq) =
        let builder = StringBuilder()
        builder.AppendLine("open StackProcessing") |> ignore
        builder.AppendLine() |> ignore
        builder.AppendLine("let availableMemory = 8UL * 1024UL * 1024UL * 1024UL") |> ignore
        builder.AppendLine() |> ignore

        states
        |> Seq.map elementLine
        |> Seq.iter (fun line -> builder.AppendLine(line) |> ignore)

        builder.ToString().TrimEnd()
