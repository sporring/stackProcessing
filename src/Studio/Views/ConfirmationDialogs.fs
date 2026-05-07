namespace Studio.Views

open System.Threading.Tasks
open Avalonia
open Avalonia.Controls
open Avalonia.Layout
open Avalonia.Media

module ConfirmationDialogs =
    let confirmAsync (owner: Window) (title: string) (message: string) =
        task {
            let dialog = Window()
            dialog.Title <- title
            dialog.Width <- 460.
            dialog.Height <- 190.
            dialog.WindowStartupLocation <- WindowStartupLocation.CenterOwner
            dialog.CanResize <- false

            let text =
                TextBlock(
                    Text = message,
                    TextWrapping = TextWrapping.Wrap,
                    Margin = Thickness(16.))

            let cancel =
                Button(
                    Content = "Cancel",
                    Width = 88.,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Margin = Thickness(0., 0., 8., 16.),
                    IsCancel = true)

            let confirm =
                Button(
                    Content = "Continue",
                    Width = 96.,
                    HorizontalAlignment = HorizontalAlignment.Right,
                    Margin = Thickness(0., 0., 16., 16.),
                    IsDefault = true)

            cancel.Click.Add(fun _ -> dialog.Close(false))
            confirm.Click.Add(fun _ -> dialog.Close(true))

            let buttons = StackPanel(Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right)
            buttons.Children.Add(cancel)
            buttons.Children.Add(confirm)

            let panel = DockPanel(LastChildFill = true)
            DockPanel.SetDock(buttons, Dock.Bottom)
            panel.Children.Add(buttons)
            panel.Children.Add(text)
            dialog.Content <- panel

            if isNull owner then
                dialog.Show()
                return false
            else
                return! dialog.ShowDialog<bool>(owner)
        }
