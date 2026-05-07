namespace Studio.Views

open Avalonia
open Avalonia.Controls
open Avalonia.Markup.Xaml
open Studio.ViewModels

type MainWindow () as this = 
    inherit Window ()

    let mutable closeConfirmed = false

    do this.InitializeComponent()
       this.Closing.Add(fun args ->
           match this.DataContext with
           | :? MainWindowViewModel as viewModel when viewModel.HasGraph && viewModel.IsGraphDirty && not closeConfirmed ->
               args.Cancel <- true

               task {
                   let! confirmed =
                       ConfirmationDialogs.confirmAsync
                           this
                           "Close Studio?"
                           "The graph currently in memory has not been saved. Close anyway?"

                   if confirmed then
                       closeConfirmed <- true
                       this.Close()
               }
               |> ignore
           | _ -> ())

    member private this.InitializeComponent() =
        AvaloniaXamlLoader.Load(this)
