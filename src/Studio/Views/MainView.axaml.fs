namespace Studio.Views

open System
open Avalonia.Controls
open Avalonia.Controls.Primitives
open Avalonia.Input
open Avalonia.Interactivity
open Avalonia.Threading
open Avalonia.Markup.Xaml
open Avalonia.VisualTree
open NodeEditor.Controls
open NodeEditor.Mvvm
open Studio.Models

type MainView() as this =
    inherit UserControl()
    do
        this.InitializeComponent()
        this.Loaded.Add(fun _ ->
            Dispatcher.UIThread.Post(fun () ->
                let editor = this.FindControl<Editor>("PipelineEditor")
                if not (isNull editor) && not (isNull editor.ZoomControl) then
                    editor.ZoomControl.FitCanvasCommand()

                if not (isNull editor) then
                    let selectCurrentNode () =
                        match editor.DrawingSource with
                        | :? DrawingNodeViewModel as drawing ->
                            drawing.GetSelectedNodes()
                            |> Seq.tryHead
                            |> Option.iter (fun node ->
                                match node.Content with
                                | :? PipelineNodeContent as nodeContent -> nodeContent.Select()
                                | _ -> ())
                        | _ -> ()

                    editor.AddHandler(
                        InputElement.PointerReleasedEvent,
                        EventHandler<PointerReleasedEventArgs>(fun _ _ ->
                            Dispatcher.UIThread.Post(selectCurrentNode)),
                        RoutingStrategies.Bubble,
                        true)

                    editor.GetVisualDescendants()
                    |> Seq.iter (fun visual ->
                        match visual with
                        | :? ScrollViewer as scrollViewer ->
                            scrollViewer.HorizontalScrollBarVisibility <- ScrollBarVisibility.Disabled
                            scrollViewer.VerticalScrollBarVisibility <- ScrollBarVisibility.Disabled
                        | :? ScrollBar as scrollBar ->
                            scrollBar.IsVisible <- false
                        | _ -> ())))

    member private this.InitializeComponent() =
        AvaloniaXamlLoader.Load(this)

    member _.PipelineNodeClicked(sender: obj, args: RoutedEventArgs) =
        match sender with
        | :? Control as control ->
            match control.DataContext with
            | :? PipelineNodeContent as nodeContent ->
                nodeContent.Select()
                args.Handled <- true
            | _ -> ()
        | _ -> ()
