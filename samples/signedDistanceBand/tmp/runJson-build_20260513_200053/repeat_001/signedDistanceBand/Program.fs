open StackProcessing

debug 1u (optimizerEnabled ()) 2147483648UL
|> readRange<uint8> 0u 1 63u "../data/rotatingBoxes" ".tiff"
>=> imageDivScalar 255uy
>=> signedDistanceBand 8u 4u
>=> cast<float,float32>
>=> write "../tmp/signedDistanceBand" ".tiff"
|> sink