namespace Studio.Models

open Avalonia.Media
open CommunityToolkit.Mvvm.ComponentModel
open Studio.Graph
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

type PipelineParameterViewModel(label: string, key: string, value: string, parameterType: BasicType, ?options: ParameterOptionViewModel list, ?canUseInput: bool, ?forceUseInput: bool, ?initialUseInput: bool) =
    inherit ObservableObject()

    let mutable value = value
    let mutable label = label
    let forceUseInput = defaultArg forceUseInput false
    let mutable useInput = forceUseInput || defaultArg initialUseInput false
    let mutable isValueEnabled = true
    let options = ObservableCollection<ParameterOptionViewModel>(defaultArg options [])
    let canUseInput = defaultArg canUseInput true

    member this.Label
        with get () = label
        and set v = this.SetProperty(&label, v) |> ignore
    member _.Key = key
    member _.ParameterType = parameterType
    member _.Options = options
    member _.HasOptions = options.Count > 0
    member _.UseBooleanEditor = parameterType = BasicType.Bool && options.Count = 0
    member this.ShowTextEditor = not this.HasOptions && not this.UseBooleanEditor
    member _.CanUseInput = canUseInput && not forceUseInput

    member this.Value
        with get () = value
        and set v =
            if this.SetProperty(&value, v) then
                this.OnPropertyChanged(nameof this.SelectedOption)
                this.OnPropertyChanged(nameof this.BooleanValue)

    member this.BooleanValue
        with get () =
            value.Trim().Equals("true", System.StringComparison.OrdinalIgnoreCase)
        and set v =
            this.Value <- if v then "true" else "false"

    member this.UseInput
        with get () = useInput
        and set v =
            let next = forceUseInput || (canUseInput && v)
            if this.SetProperty(&useInput, next) then
                this.OnPropertyChanged(nameof this.IsValueEditable)
                this.OnPropertyChanged(nameof this.IsValueReadOnly)

    member _.IsValueEditable = not useInput && not forceUseInput
    member this.IsValueReadOnly = not this.IsValueEditable

    member this.IsValueEnabled
        with get () = isValueEnabled
        and set v = this.SetProperty(&isValueEnabled, v) |> ignore

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
    let mutable isProblemHighlighted = false
    let mutable isSelected = false

    member _.Definition = definition
    member _.Parameters = System.Collections.ObjectModel.ObservableCollection<PipelineParameterViewModel>(parameters)
    member _.Summary = definition.Summary
    member _.Description = definition.Description
    member _.HasDescription = not (System.String.IsNullOrWhiteSpace definition.Description)

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

    member this.IsProblemHighlighted
        with get () = isProblemHighlighted
        and set v =
            if this.SetProperty(&isProblemHighlighted, v) then
                this.OnPropertyChanged(nameof this.NodeBackground)
                this.OnPropertyChanged(nameof this.NodeBorderBrush)

    member this.IsSelected
        with get () = isSelected
        and set v =
            if this.SetProperty(&isSelected, v) then
                this.OnPropertyChanged(nameof this.NodeBorderBrush)

    member this.NodeBackground =
        if isPaletteDragOutside then
            SolidColorBrush.Parse("#FFF8E1") :> IBrush
        elif isProblemHighlighted then
            SolidColorBrush.Parse("#FFE5E5") :> IBrush
        elif definition.Id = "ComputeStats" || definition.Id = "ComponentTranslationTable" || definition.Id = "HistogramData" then
            SolidColorBrush.Parse("#F3E0C3") :> IBrush
        else
            SolidColorBrush.Parse("#EAF3FF") :> IBrush

    member this.NodeBorderBrush =
        if isSelected then
            SolidColorBrush.Parse("#0B5CAD") :> IBrush
        elif isPaletteDragOutside then
            SolidColorBrush.Parse("#F2A900") :> IBrush
        elif isProblemHighlighted then
            SolidColorBrush.Parse("#D64545") :> IBrush
        elif definition.Id = "ComputeStats" || definition.Id = "ComponentTranslationTable" || definition.Id = "HistogramData" then
            SolidColorBrush.Parse("#B7791F") :> IBrush
        else
            SolidColorBrush.Parse("#2F80ED") :> IBrush

    member this.NodeOpacity =
        if isPaletteDragOutside then 0.0 else 1.0

type PipelineNodeContent(label: string, state: PipelineNodeState, width: float, height: float, select: unit -> unit) =
    member _.Label = label
    member _.State = state
    member _.Width = width
    member _.Height = height
    member _.Select() = select()

type PaletteGroupViewModel(title: string, functions: Function seq, isExpanded: bool) =
    member _.Title = title
    member _.Functions = ObservableCollection<Function>(functions)
    member _.IsExpanded = isExpanded

type IGraphWindowController =
    abstract member SetDrawingSize: width: float -> height: float -> unit
    abstract member MoveSelectionBy: dx: float -> dy: float -> unit
    abstract member DeleteSelectedElementIfInTrashZone: trashWidth: float -> trashHeight: float -> margin: float -> unit
