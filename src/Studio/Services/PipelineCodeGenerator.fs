namespace Studio.Services

open System
open System.Collections.ObjectModel
open System.Text
open Studio.Models

module PipelineCodeGenerator =

    let private paramValue key (element: PipelineElementViewModel) =
        element.Parameters
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

    let private elementLine (element: PipelineElementViewModel) =
        match element.Kind with
        | PipelineElementKind.Source ->
            let availableMemory = paramValue "availableMemory" element
            $"source {availableMemory}"
        | PipelineElementKind.Read ->
            let pixelType = paramValue "pixelType" element
            let input = paramValue "input" element |> quote
            let suffix = paramValue "suffix" element |> quote
            $"|> read<{pixelType}> {input} {suffix}"
        | PipelineElementKind.DiscreteGaussian ->
            let sigma = paramValue "sigma" element
            let outputRegionMode = paramValue "outputRegionMode" element |> optionRaw
            let boundaryCondition = paramValue "boundaryCondition" element |> optionRaw
            let windowSize = paramValue "windowSize" element |> optionUInt
            $">=> discreteGaussian {sigma} {outputRegionMode} {boundaryCondition} {windowSize}"
        | PipelineElementKind.Cast ->
            let sourceType = paramValue "sourceType" element
            let targetType = paramValue "targetType" element
            $">=> cast<{sourceType},{targetType}>"
        | PipelineElementKind.Write ->
            let output = paramValue "output" element |> quote
            let suffix = paramValue "suffix" element |> quote
            $">=> write {output} {suffix}"
        | PipelineElementKind.Sink ->
            "|> sink"
        | _ ->
            $"// Unsupported element: {element.Title}"

    let generate (elements: ObservableCollection<PipelineElementViewModel>) =
        let builder = StringBuilder()
        builder.AppendLine("open StackProcessing") |> ignore
        builder.AppendLine() |> ignore
        builder.AppendLine("let availableMemory = 8UL * 1024UL * 1024UL * 1024UL") |> ignore
        builder.AppendLine() |> ignore

        elements
        |> Seq.map elementLine
        |> Seq.iter (fun line -> builder.AppendLine(line) |> ignore)

        builder.ToString().TrimEnd()
