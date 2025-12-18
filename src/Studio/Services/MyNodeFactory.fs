namespace Studio.Services

open System.Collections.Generic
open System.Collections.ObjectModel
open NodeEditor.Model
open NodeEditor.Mvvm

type MyNodeFactory() =

    let createSimpleNode () =
        let node = NodeViewModel()
        node.Name <- "HELLO"
        node.X <- 0
        node.Y <- 0
        node.Width <- 160
        node.Height <- 60
        node.Pins <- ObservableCollection<IPin>()

        // one input on left, one output on right
        node.AddPin(0., 30., 10., 10., PinAlignment.Left, "IN") |> ignore
        node.AddPin(160., 30., 10., 10., PinAlignment.Right, "OUT") |> ignore

        node :> INode

    interface INodeFactory with

        member _.CreateDrawing(name:string) =
            let settings = DrawingNodeSettingsViewModel()
            settings.EnableMultiplePinConnections <- true
            settings.EnableSnap <- true
            settings.SnapX <- 15.
            settings.SnapY <- 15.
            settings.EnableGrid <- true
            settings.GridCellWidth <- 15.
            settings.GridCellHeight <- 15.

            let drawing = DrawingNodeViewModel()
            drawing.Settings <- settings
            drawing.Name <- name
            drawing.X <- 0.
            drawing.Y <- 0.
            drawing.Width <- 1200.
            drawing.Height <- 800.
            drawing.Nodes <- ObservableCollection<INode>()
            drawing.Connectors <- ObservableCollection<IConnector>()
            drawing :> IDrawingNode

        member _.CreateTemplates() =
            let node = createSimpleNode()
            let template = NodeTemplateViewModel()
            template.Title <- "Hello Node"
            template.Template <- node
            template.Preview <- createSimpleNode()
            ObservableCollection<INodeTemplate>([ template :> INodeTemplate ]) :> IList<INodeTemplate>
