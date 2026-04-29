namespace Studio.Models

open CommunityToolkit.Mvvm.ComponentModel

type PipelineElementKind =
    | Source = 0
    | Read = 1
    | DiscreteGaussian = 2
    | Cast = 3
    | Write = 4
    | Sink = 5

type PipelineParameterViewModel(label: string, key: string, value: string) =
    inherit ObservableObject()

    let mutable value = value

    member _.Label = label
    member _.Key = key

    member this.Value
        with get () = value
        and set v = this.SetProperty(&value, v) |> ignore

[<AllowNullLiteral>]
type PipelineElementViewModel(kind: PipelineElementKind, title: string, parameters: PipelineParameterViewModel list) =
    inherit ObservableObject()

    let mutable title = title

    member _.Kind = kind
    member _.Parameters = System.Collections.ObjectModel.ObservableCollection<PipelineParameterViewModel>(parameters)

    member this.Title
        with get () = title
        and set v = this.SetProperty(&title, v) |> ignore

type PipelineNodeContent(label: string, select: unit -> unit) =
    member _.Label = label
    member _.Select() = select()
