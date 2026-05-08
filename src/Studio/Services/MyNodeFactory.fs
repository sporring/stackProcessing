namespace Studio.Services

open System.Collections.Generic
open System.Collections.ObjectModel
open NodeEditor.Model
open NodeEditor.Mvvm

type MyNodeFactory() =

    let createSimpleNode title =
        let node = NodeViewModel()
        node.Name <- title
        node.X <- 0
        node.Y <- 0
        node.Width <- 110
        node.Height <- 48
        node.Pins <- ObservableCollection<IPin>()

        node :> INode

    interface INodeFactory with

        member _.CreateDrawing(name:string) =
            let settings = DrawingNodeSettingsViewModel()
            settings.EnableMultiplePinConnections <- true
            settings.EnableSnap <- true
            settings.SnapX <- 15.
            settings.SnapY <- 15.
            settings.EnableGrid <- false
            settings.GridCellWidth <- 15.
            settings.GridCellHeight <- 15.

            let drawing = DrawingNodeViewModel()
            drawing.Settings <- settings
            drawing.Name <- name
            drawing.X <- 0.
            drawing.Y <- 0.
            drawing.Width <- 760.
            drawing.Height <- 180.
            drawing.Nodes <- ObservableCollection<INode>()
            drawing.Connectors <- ObservableCollection<IConnector>()
            drawing :> IDrawingNode

        member _.CreateTemplates() =
            let template title =
                let template = NodeTemplateViewModel()
                template.Title <- title
                template.Template <- createSimpleNode title
                template.Preview <- createSimpleNode title
                template :> INodeTemplate

            ObservableCollection<INodeTemplate>(
                [ template "source"
                  template "read"
                  template "smoothWGauss"
                  template "cast"
                  template "write"
                  template "sink" ])
            :> IList<INodeTemplate>
