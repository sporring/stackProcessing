namespace Graph

open System.IO
open System.Text
open System.Text.Json
open System.Threading.Tasks

module PipelineGraphStorage =
    let private jsonOptions =
        JsonSerializerOptions(WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    let serialize (graph: SavedGraph) =
        JsonSerializer.Serialize(graph, jsonOptions)

    let deserialize (json: string) =
        let graph = JsonSerializer.Deserialize<SavedGraph>(json, jsonOptions)

        if isNull (box graph) then
            invalidOp "The selected file did not contain a pipeline graph."

        graph

    let writeJsonAsync (stream: Stream) (graph: SavedGraph) =
        task {
            if stream.CanSeek then
                stream.SetLength(0L)

            use writer = new StreamWriter(stream, Encoding.UTF8, 1024, leaveOpen = true)
            do! writer.WriteAsync(serialize graph)
            do! writer.FlushAsync()
        }
        :> Task

    let readJsonAsync (stream: Stream) =
        task {
            use reader = new StreamReader(stream, Encoding.UTF8, true, 1024, leaveOpen = true)
            let! json = reader.ReadToEndAsync()
            return deserialize json
        }

    let save (path: string) (graph: SavedGraph) =
        File.WriteAllText(path, serialize graph)

    let load (path: string) =
        File.ReadAllText(path) |> deserialize
