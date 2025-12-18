namespace Studio.ViewModels

open System.Collections.ObjectModel
open NodeEditor.Mvvm
open Studio.Services

type MainWindowViewModel() =

    member val Editor : EditorViewModel = 
        let editor = EditorViewModel()
        //editor.Serializer <- NodeSerializer(typeof<ObservableCollection<_>>)
        editor.Factory <- MyNodeFactory()
        editor.Templates <- editor.Factory.CreateTemplates()
        editor.Drawing <- editor.Factory.CreateDrawing("My Pipeline")
        //editor.Drawing.SetSerializer(editor.Serializer)
        editor
        with get, set
