module StackCharts

open Plotly.NET

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

let showImageWithLabels colorMap title xAxis yAxis image =
    Chart.Heatmap(ImageFunctions.toSeqSeq image, ColorScale = colorScale colorMap)
    |> applyChartLabels title xAxis yAxis
    |> Chart.show

let showImage image =
    showImageWithLabels "Viridis" "" "" "" image
