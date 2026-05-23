module ProbeSelection

open System

type EvidenceSelector =
    { Families: string list
      Members: string list
      UpTo: string option }

let ladder =
    [ "io"
      "io-cast"
      "sources"
      "singleton"
      "neighbourhood"
      "geometry"
      "fourier"
      "keypoints"
      "dependency"
      "reducers" ]

let splitCsvList (value: string) =
    value.Split([| ','; ';' |], StringSplitOptions.RemoveEmptyEntries ||| StringSplitOptions.TrimEntries)
    |> Array.toList

let normalizeFamily (value: string) =
    match value.Trim().ToLowerInvariant().Replace("_", "-") with
    | "all" -> Some "all"
    | "io" | "read-write" | "readwrite" -> Some "io"
    | "io-cast" | "io-casts" | "read-cast" | "readcast" | "conversion" | "conversions" -> Some "io-cast"
    | "sources" | "source" -> Some "sources"
    | "singleton" | "singletons" | "simple" | "simple-unary" -> Some "singleton"
    | "neighbourhood" | "neighborhood" | "window" | "windowed" | "windowed-unary" -> Some "neighbourhood"
    | "geometry" | "projection" | "geometry-and-projection" -> Some "geometry"
    | "fourier" | "vector" | "fourier-and-vector" -> Some "fourier"
    | "keypoints" | "keypoint" | "distance" | "keypoint-and-distance" -> Some "keypoints"
    | "dependency" | "dependency-breakers" -> Some "dependency"
    | "reducers" | "reducer" -> Some "reducers"
    | _ -> None

let parseFamilies value =
    let tokens = splitCsvList value
    if tokens.IsEmpty then
        None
    else
        let parsed = tokens |> List.map normalizeFamily
        if parsed |> List.forall Option.isSome then
            parsed |> List.choose id |> List.distinct |> Some
        else
            None

let familiesUpTo maxStep =
    match normalizeFamily maxStep with
    | Some "all" -> Some ladder
    | Some family ->
        ladder
        |> List.tryFindIndex ((=) family)
        |> Option.map (fun index -> ladder |> List.take (index + 1))
    | None -> None

let familyForRowId (rowId: string) =
    let text = rowId.ToLowerInvariant()
    if text.Contains("01-starters") then Some "io"
    elif text.Contains("02-io-casts") then Some "io-cast"
    elif text.Contains("02-sources") then Some "sources"
    elif text.Contains("03-simple-unary") || text.Contains("05-intensity-and-additive") then Some "singleton"
    elif text.Contains("04-windowed-unary") then Some "neighbourhood"
    elif text.Contains("06-geometry-and-projection") then Some "geometry"
    elif text.Contains("07-fourier-and-vector") then Some "fourier"
    elif text.Contains("08-keypoint-and-distance") then Some "keypoints"
    elif text.Contains("09-dependency-breakers") then Some "dependency"
    elif text.Contains("10-reducers") then Some "reducers"
    else None

let normalizeMember (value: string) =
    value.Trim().ToLowerInvariant()

let selectedFamilies selector =
    match selector.UpTo with
    | Some upTo ->
        familiesUpTo upTo
        |> Option.defaultValue selector.Families
    | None ->
        if selector.Families.IsEmpty || selector.Families |> List.exists ((=) "all") then
            ladder
        else
            selector.Families

let selectorMatchesFamily selector family =
    selectedFamilies selector |> List.contains family

let selectorMatchesMembers selector members =
    if selector.Members.IsEmpty then
        true
    else
        let memberSet = selector.Members |> List.map normalizeMember |> Set.ofList
        members |> List.exists (normalizeMember >> memberSet.Contains)
