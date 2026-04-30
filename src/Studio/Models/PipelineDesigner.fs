namespace Studio.Models

open CommunityToolkit.Mvvm.ComponentModel
open Graph
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
type PipelineNodeState(definition: FunctionDefinition, parameters: PipelineParameterViewModel list) =
    inherit ObservableObject()

    let mutable title = definition.DisplayName

    member _.Kind =
        match System.Enum.TryParse<PipelineElementKind>(definition.Id) with
        | true, kind -> kind
        | _ -> invalidOp $"Unsupported legacy pipeline element kind: {definition.Id}"

    member _.Definition = definition
    member _.Parameters = System.Collections.ObjectModel.ObservableCollection<PipelineParameterViewModel>(parameters)

    member this.Title
        with get () = title
        and set v = this.SetProperty(&title, v) |> ignore

type PipelineNodeContent(label: string, state: PipelineNodeState, select: unit -> unit) =
    member _.Label = label
    member _.State = state
    member _.Select() = select()

type PaletteGroupViewModel(title: string, functions: FunctionDefinition seq, isExpanded: bool) =
    member _.Title = title
    member _.Functions = ObservableCollection<FunctionDefinition>(functions)
    member _.IsExpanded = isExpanded

type IGraphWindowController =
    abstract member SetDrawingSize: width: float -> height: float -> unit
    abstract member DeleteSelectedElementIfInTrashZone: trashWidth: float -> trashHeight: float -> margin: float -> unit
