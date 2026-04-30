namespace Studio.Models

open CommunityToolkit.Mvvm.ComponentModel
open System.Collections.ObjectModel

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

type PipelineNodeContent(label: string, element: PipelineElementViewModel, select: unit -> unit) =
    member _.Label = label
    member _.Element = element
    member _.Select() = select()

type PaletteFunctionViewModel(kind: PipelineElementKind, name: string, category: string, description: string, aliases: string list) =
    member _.Kind = kind
    member _.KindName = kind.ToString()
    member _.Name = name
    member _.Category = category
    member _.Description = description
    member _.Aliases = aliases

    member this.Matches(searchText: string) =
        let contains (value: string) =
            value.Contains(searchText, System.StringComparison.OrdinalIgnoreCase)

        System.String.IsNullOrWhiteSpace(searchText)
        || contains this.Name
        || contains this.Category
        || contains this.Description
        || (this.Aliases |> List.exists contains)

type PaletteGroupViewModel(title: string, functions: PaletteFunctionViewModel seq, isExpanded: bool) =
    member _.Title = title
    member _.Functions = ObservableCollection<PaletteFunctionViewModel>(functions)
    member _.IsExpanded = isExpanded

type IGraphWindowController =
    abstract member SetDrawingSize: width: float -> height: float -> unit
    abstract member DeleteSelectedElementIfInTrashZone: trashWidth: float -> trashHeight: float -> margin: float -> unit
