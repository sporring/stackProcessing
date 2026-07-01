module StackCharts

open System
open Plotly.NET
open StackCore

let private applyChartLabels title xAxis yAxis chart =
    let withTitle =
        if System.String.IsNullOrWhiteSpace title then chart
        else chart |> Chart.withTitle title

    let withXAxis =
        if System.String.IsNullOrWhiteSpace xAxis then withTitle
        else withTitle |> Chart.withXAxisStyle xAxis

    if System.String.IsNullOrWhiteSpace yAxis then withXAxis
    else withXAxis |> Chart.withYAxisStyle yAxis

let private colorScale name =
    match name with
    | "Blackbody" -> StyleParam.Colorscale.Blackbody
    | "Bluered" -> StyleParam.Colorscale.Bluered
    | "Cividis" -> StyleParam.Colorscale.Cividis
    | "Earth" -> StyleParam.Colorscale.Earth
    | "Electric" -> StyleParam.Colorscale.Electric
    | "Greens" -> StyleParam.Colorscale.Greens
    | "Greys" -> StyleParam.Colorscale.Greys
    | "Hot" -> StyleParam.Colorscale.Hot
    | "Jet" -> StyleParam.Colorscale.Jet
    | "Picnic" -> StyleParam.Colorscale.Picnic
    | "Portland" -> StyleParam.Colorscale.Portland
    | "Rainbow" -> StyleParam.Colorscale.Rainbow
    | "RdBu" -> StyleParam.Colorscale.RdBu
    | "YlGnBu" -> StyleParam.Colorscale.YIGnBu
    | "YlOrRd" -> StyleParam.Colorscale.YIOrRd
    | _ -> StyleParam.Colorscale.Viridis

let chartData kind x y =
    match kind with
    | "Scatter" -> Chart.Scatter(x = x, y = y, mode = StyleParam.Mode.Markers)
    | "Line" -> Chart.Line(x = x, y = y)
    | "Bar" -> Chart.Bar(values = y, Keys = x)
    | "Area" -> Chart.Area(x = x, y = y)
    | "Pie" -> Chart.Pie(values = y, Labels = x)
    | "Doughnut" -> Chart.Doughnut(values = y, Labels = x)
    | _ -> Chart.Column(values = y, Keys = x)

let showChartDataWithLabels kind title xAxis yAxis x y =
    chartData kind x y
    |> applyChartLabels title xAxis yAxis
    |> Chart.show

let showChartData kind x y =
    showChartDataWithLabels kind "" "" "" x y

let showChartWithLabels kind title xAxis yAxis points =
    let x, y = points |> Map.toList |> List.unzip
    showChartDataWithLabels kind title xAxis yAxis x y

let showChart kind points =
    showChartWithLabels kind "" "" "" points

let showChartXYWithLabels kind title xAxis yAxis x y =
    showChartDataWithLabels kind title xAxis yAxis x y

let showChartXY kind x y =
    showChartXYWithLabels kind "" "" "" x y

let private histogramToXY (histogram: Histogram<'T>) =
    histogram.Counts
    |> Map.toList
    |> List.map (fun (x, y) -> Convert.ToDouble x, Convert.ToDouble y)
    |> List.unzip

let showHistogramWithLabels title xAxis yAxis histogram =
    let x, y = histogramToXY histogram
    showChartDataWithLabels "Column" title xAxis yAxis x y

let showHistogram histogram =
    showHistogramWithLabels "" "" "" histogram

let private chunkToSeqSeq<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    let width64, height64, depth64 = chunk.Size
    if depth64 <> 1UL then
        invalidArg "chunk" $"showChunk expects 2D slice chunks with depth 1, got {chunk.Size}."
    if width64 > uint64 Int32.MaxValue || height64 > uint64 Int32.MaxValue then
        invalidArg "chunk" $"showChunk dimensions must fit Int32, got {chunk.Size}."

    let width = int width64
    let height = int height64
    let pixels = Chunk.span chunk
    let rows = Array.zeroCreate<float[]> height
    for y in 0 .. height - 1 do
        let row = Array.zeroCreate<float> width
        for x in 0 .. width - 1 do
            row[x] <- Convert.ToDouble(pixels[Chunk.toIndex width height x y 0])
        rows[y] <- row
    rows |> Seq.map (fun row -> row :> seq<float>)

let showChunkWithLabels<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> colorMap title xAxis yAxis (chunk: Chunk<'T>) =
    Chart.Heatmap(chunkToSeqSeq chunk, ColorScale = colorScale colorMap)
    |> applyChartLabels title xAxis yAxis
    |> Chart.show

let showChunk<'T when 'T: equality and 'T: (new: unit -> 'T) and 'T: struct and 'T :> ValueType> (chunk: Chunk<'T>) =
    showChunkWithLabels "Viridis" "" "" "" chunk
