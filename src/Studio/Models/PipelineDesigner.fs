namespace Studio.Models

open Avalonia.Media
open CommunityToolkit.Mvvm.ComponentModel
open Graph
open System.Collections.ObjectModel

[<AllowNullLiteral>]
type ParameterOptionViewModel(label: string, value: string, isEnabled: bool) =
    inherit ObservableObject()

    let mutable isEnabled = isEnabled

    member _.Label = label
    member _.Value = value

    member this.IsEnabled
        with get () = isEnabled
        and set v = this.SetProperty(&isEnabled, v) |> ignore

type PipelineParameterViewModel(label: string, key: string, value: string, parameterType: BasicType, ?options: ParameterOptionViewModel list, ?canUseInput: bool) =
    inherit ObservableObject()

    let mutable value = value
    let mutable useInput = false
    let options = ObservableCollection<ParameterOptionViewModel>(defaultArg options [])
    let canUseInput = defaultArg canUseInput true

    member _.Label = label
    member _.Key = key
    member _.ParameterType = parameterType
    member _.Options = options
    member _.HasOptions = options.Count > 0
    member _.CanUseInput = canUseInput

    member this.Value
        with get () = value
        and set v =
            if this.SetProperty(&value, v) then
                this.OnPropertyChanged(nameof this.SelectedOption)

    member this.UseInput
        with get () = useInput
        and set v =
            let next = canUseInput && v
            if this.SetProperty(&useInput, next) then
                this.OnPropertyChanged(nameof this.IsValueEditable)

    member _.IsValueEditable = canUseInput && not useInput

    member this.SelectedOption
        with get () =
            options
            |> Seq.tryFind (fun option -> option.Value = value)
            |> Option.toObj
        and set (option: ParameterOptionViewModel) =
            if not (isNull option) && option.IsEnabled then
                this.Value <- option.Value

[<AllowNullLiteral>]
type PipelineNodeState(definition: Function, parameters: PipelineParameterViewModel list) =
    inherit ObservableObject()

    let mutable title = definition.DisplayName
    let mutable isPaletteDragOutside = false

    member _.Definition = definition
    member _.Parameters = System.Collections.ObjectModel.ObservableCollection<PipelineParameterViewModel>(parameters)

    member this.Title
        with get () = title
        and set v = this.SetProperty(&title, v) |> ignore

    member this.IsPaletteDragOutside
        with get () = isPaletteDragOutside
        and set v =
            if this.SetProperty(&isPaletteDragOutside, v) then
                this.OnPropertyChanged(nameof this.NodeBackground)
                this.OnPropertyChanged(nameof this.NodeBorderBrush)
                this.OnPropertyChanged(nameof this.NodeOpacity)

    member this.NodeBackground =
        if isPaletteDragOutside then
            SolidColorBrush.Parse("#FFF8E1") :> IBrush
        else
            SolidColorBrush.Parse("#EAF3FF") :> IBrush

    member this.NodeBorderBrush =
        if isPaletteDragOutside then
            SolidColorBrush.Parse("#F2A900") :> IBrush
        else
            SolidColorBrush.Parse("#2F80ED") :> IBrush

    member this.NodeOpacity =
        if isPaletteDragOutside then 0.0 else 1.0

type PipelineNodeContent(label: string, state: PipelineNodeState, select: unit -> unit) =
    member _.Label = label
    member _.State = state
    member _.Select() = select()

type PaletteGroupViewModel(title: string, functions: Function seq, isExpanded: bool) =
    member _.Title = title
    member _.Functions = ObservableCollection<Function>(functions)
    member _.IsExpanded = isExpanded

type IGraphWindowController =
    abstract member SetDrawingSize: width: float -> height: float -> unit
    abstract member DeleteSelectedElementIfInTrashZone: trashWidth: float -> trashHeight: float -> margin: float -> unit
