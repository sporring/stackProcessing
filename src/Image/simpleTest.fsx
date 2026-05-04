#r "bin/Debug/net10.0/Image.dll"            
#r "bin/Debug/net10.0/SimpleITKCSharpManaged.dll"
open Image
open ImageFunctions

let makeImpulse2D () =
    Array2D.init 5 5 (fun m n -> if m = 2 && n = 2 then 1.0 else 0.0)

do
    let scalarImage = Image<int>([10u; 12u])
    let vectorImage = Image<int list>([10u; 12u])

    printfn "%A" scalarImage[1, 2]
    scalarImage[1, 2] <- 1
    printfn "%A" scalarImage[1, 2]

    printfn "%A" vectorImage[2, 1]
    vectorImage[1, 2] <- [2; 1]
    printfn "%A" vectorImage[1, 2]

    let squeezedSource = Image<int list>([10u; 1u; 12u])
    printfn "%A" squeezedSource

    let squeezed = ImageFunctions.squeeze squeezedSource
    printfn "%A" squeezedSource
    printfn "%A" squeezed

    let shifted = ImageFunctions.imageAddScalar scalarImage 5
    printfn "%A" (ImageFunctions.concatAlong 0u scalarImage shifted)
    printfn "%A" (ImageFunctions.concatAlong 1u scalarImage shifted)
    printfn "%A" (ImageFunctions.concatAlong 2u scalarImage shifted)
    printfn "%A" (ImageFunctions.concatAlong 3u scalarImage shifted)

do
    let sourceArray = array2D [[10.0f; 20.0f]; [30.0f; 40.0f]]
    let floatImage = Image<float32>.ofArray2D sourceArray
    let byteImage = floatImage.castTo<uint8>()
    printfn "%A" byteImage

do
    let image = Image<float>.ofArray2D (array2D [[1.0; 2.0]; [3.0; 4.0]])
    let kernel = Image<float>.ofArray2D (array2D [[0.0; 0.0; 0.0]; [0.0; 1.0; 0.0]; [0.0; 0.0; 0.0]])
    let result = ImageFunctions.conv image kernel
    printfn "%A" result

do
    let image = Image<float>.ofArray2D (makeImpulse2D ())
    let blurred = ImageFunctions.discreteGaussian 2u 1.0 None None None image
    printfn "%A" blurred

do
    let image = Image<float>.ofArray2D (makeImpulse2D ())
    let blurredValid = ImageFunctions.discreteGaussian 2u 1.0 None (Some ImageFunctions.Valid) None image
    printfn "%A" blurredValid

do
    let image = Image<float>.ofArray2D (makeImpulse2D ())
    let gradient = ImageFunctions.gradientConvolve 0u 1u image
    printfn "%A" gradient

do
    let image = Image<uint8>.ofArray2D (array2D [[0uy; 1uy; 1uy; 0uy]])
    let eroded = ImageFunctions.binaryErode 1u image
    printfn "%A" eroded

do
    let image = Image<uint8>.ofArray2D (array2D [[0uy; 1uy; 0uy]])
    let dilated = ImageFunctions.binaryDilate 1u image
    printfn "%A" dilated

do
    let image = Image<uint8>.ofArray2D (array2D [[0uy; 1uy; 0uy; 1uy; 0uy]])
    let opened = ImageFunctions.binaryOpening 1u image
    printfn "%A" opened

do
    let image = Image<uint8>.ofArray2D (array2D [[1uy; 0uy; 1uy]])
    let closed = ImageFunctions.binaryClosing 1u image
    printfn "%A" closed

do
    let image = Image<uint8>.ofArray2D (array2D [[1uy; 1uy; 1uy]; [1uy; 0uy; 1uy]; [1uy; 1uy; 1uy]])
    let filled = ImageFunctions.binaryFillHoles image
    printfn "%A" filled

do
    let image = Image<uint8>.ofArray2D (array2D [[1uy; 0uy; 1uy]])
    let labels = (ImageFunctions.connectedComponents image).Labels
    printfn "%A" labels

do
    let image = Image<uint8>.ofArray2D (array2D [[1uy; 0uy; 1uy]])
    let labels = (ImageFunctions.connectedComponents image).Labels
    let relabeled = ImageFunctions.relabelComponents 2u labels
    printfn "%A" relabeled

do
    let image = Image<uint8>.ofArray2D (array2D [[0uy; 1uy; 0uy]])
    let distanceMap = ImageFunctions.signedDistanceMap 1uy 0uy image
    printfn "%A" distanceMap

do
    let image = Image<uint8>.ofArray2D (array2D [[0uy; 1uy; 0uy; 1uy; 0uy]])
    let watershed = ImageFunctions.watershed 0.0 image
    printfn "%A" watershed
