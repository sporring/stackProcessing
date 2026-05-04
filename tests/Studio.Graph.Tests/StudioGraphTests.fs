module Tests.StudioGraphTests

open System.IO
open System.Text
open Expecto
open Studio.Graph

let private savedParam key value useInput =
    { Key = key; Value = value; UseInput = useInput }

let private savedNode id functionId parameters =
    { Id = id
      FunctionId = functionId
      X = 12.5
      Y = -7.0
      Parameters = parameters |> List.toArray }

[<Tests>]
let domainSuite =
    testList "Studio.Graph domain" [
        testCase "numeric and basic type strings roundtrip" <| fun _ ->
            let numericTypes =
                [ Number; UInt8; Int8; UInt16; Int16; UInt32; Int32; UInt64; Int64; Float32; Float64; Complex ]

            for numericType in numericTypes do
                let text = NumericType.toString numericType
                Expect.equal (NumericType.tryParse text) (Some numericType) $"NumericType should parse {text}."
                Expect.equal (BasicType.tryParse text) (Some(BasicType.Numeric numericType)) $"BasicType should parse numeric {text}."

            Expect.equal (BasicType.tryParse "String") (Some BasicType.String) "String should parse."
            Expect.equal (BasicType.tryParse "Histogram") (Some BasicType.Map) "Histogram is currently stored as Map."
            Expect.equal (BasicType.tryParse "nope") None "Unknown type should not parse."

        testCase "port compatibility supports Any and Number wildcards" <| fun _ ->
            Expect.isTrue (PortType.canConnect Any (Scalar String)) "Any output should connect to concrete input."
            Expect.isTrue (PortType.canConnect (Image UInt8) (Image Number)) "Concrete image should connect to Number image input."
            Expect.isTrue (PortType.canConnect (Image Number) (Image Float64)) "Number image should connect to concrete image input."
            Expect.isFalse (PortType.canConnect (Scalar String) (Scalar(BasicType.Numeric Float64))) "Different concrete scalar types should not connect."

        testCase "FunctionDefinition.matches searches display category description and aliases" <| fun _ ->
            let read = BuiltInCatalog.find "Read"
            Expect.isTrue (FunctionDefinition.matches "read" read) "Display name should match."
            Expect.isTrue (FunctionDefinition.matches "Sources" read) "Category should match."
            Expect.isTrue (FunctionDefinition.matches "chunked" read) "Description should match."
            Expect.isTrue (FunctionDefinition.matches "tiff" read) "Alias should match."
            Expect.isFalse (FunctionDefinition.matches "definitely-not-here" read) "Unrelated search should not match."
    ]

[<Tests>]
let catalogSuite =
    testList "Studio.Graph catalog" [
        testCase "catalog exposes expected generic functions" <| fun _ ->
            let ids = BuiltInCatalog.orderedFunctions |> List.map _.Id
            Expect.containsAll ids ["Scalar"; "Read"; "Write"; "ImageOpImage"; "ComputeStats"; "Chart"] "Important Studio functions should be in the palette catalog."

        testCase "find fails clearly for missing function" <| fun _ ->
            Expect.throws (fun () -> BuiltInCatalog.find "MissingFunction" |> ignore) "Missing function lookup should fail."
    ]

[<Tests>]
let persistenceSuite =
    testList "Studio.Graph persistence" [
        testCase "serialize deserialize roundtrips saved graph" <| fun _ ->
            let graph =
                { Version = 1
                  Nodes =
                    [| savedNode "n1" "Scalar" [ savedParam "type" "String" false; savedParam "value" "input" false ]
                       savedNode "n2" "Read" [ savedParam "input" "input" true ] |]
                  Edges =
                    [| { FromNode = "n1"; FromKind = "scalarOutput"; FromPort = 0; ToNode = "n2"; ToKind = "parameterInput"; ToPort = 0 } |] }

            let json = PipelineGraphStorage.serialize graph
            let restored = PipelineGraphStorage.deserialize json
            Expect.equal restored graph "Saved graph should survive JSON roundtrip."
            Expect.stringContains json "\"functionId\": \"Scalar\"" "JSON should use camelCase field names."

        testCase "writeJsonAsync truncates and writes stream" <| fun _ ->
            let graph =
                { Version = 2
                  Nodes = [| savedNode "scalar" "Scalar" [ savedParam "type" "Float64" false ] |]
                  Edges = [||] }

            use stream = new MemoryStream()
            let oldBytes = Encoding.UTF8.GetBytes("old text that should be removed")
            stream.Write(oldBytes, 0, oldBytes.Length)
            stream.Position <- 0L
            PipelineGraphStorage.writeJsonAsync stream graph
            |> Async.AwaitTask
            |> Async.RunSynchronously

            stream.Position <- 0L
            let restored =
                PipelineGraphStorage.readJsonAsync stream
                |> Async.AwaitTask
                |> Async.RunSynchronously

            Expect.equal restored graph "Stream roundtrip should preserve graph."
            Expect.isFalse ((Encoding.UTF8.GetString(stream.ToArray())).Contains("old text")) "Old stream contents should be truncated before writing."
    ]
