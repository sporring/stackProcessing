module Tests

open Expecto
open ImageClass  // Ensure namespace is correct
open itk.simple

[<Tests>]
let pixelTypeTests =
  testList "PixelType Tests" [
    testCase "ToSimpleITK UInt8" <| fun _ ->
        let value = PixelType.UInt8
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt8 "Expected sitkUInt8"
    
    testCase "ToSimpleITK Int8" <| fun _ ->
        let value = PixelType.Int8
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkInt8 "Expected sitkInt8"
    
    testCase "ToSimpleITK UInt16" <| fun _ ->
        let value = PixelType.UInt16
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt16 "Expected sitkUInt16"
    
    testCase "ToSimpleITK Int16" <| fun _ ->
        let value = PixelType.Int16
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkInt16 "Expected sitkInt16"
    
    testCase "ToSimpleITK UInt32" <| fun _ ->
        let value = PixelType.UInt32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt32 "Expected sitkUInt32"
    
    testCase "ToSimpleITK Int32" <| fun _ ->
        let value = PixelType.Int32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkInt32 "Expected sitkInt32"
    
    testCase "ToSimpleITK UInt64" <| fun _ ->
        let value = PixelType.UInt64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkUInt64 "Expected sitkUInt64"
    
    testCase "ToSimpleITK Int64" <| fun _ ->
        let value = PixelType.Int64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkInt64 "Expected sitkInt64"
    
    testCase "ToSimpleITK Float32" <| fun _ ->
        let value = PixelType.Float32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkFloat32 "Expected sitkFloat32"
    
    testCase "ToSimpleITK Float64" <| fun _ ->
        let value = PixelType.Float64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkFloat64 "Expected sitkFloat64"
    
    testCase "ToSimpleITK ComplexFloat32" <| fun _ ->
        let value = PixelType.ComplexFloat32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat32 "Expected sitkVectorFloat32"
    
    testCase "ToSimpleITK ComplexFloat64" <| fun _ ->
        let value = PixelType.ComplexFloat64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat64 "Expected sitkVectorFloat64"
    
    testCase "ToSimpleITK VectorUInt8" <| fun _ ->
        let value = PixelType.VectorUInt8
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt8 "Expected sitkVectorUInt8"
    
    testCase "ToSimpleITK VectorInt8" <| fun _ ->
        let value = PixelType.VectorInt8
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt8 "Expected sitkVectorInt8"
    
    testCase "ToSimpleITK VectorUInt16" <| fun _ ->
        let value = PixelType.VectorUInt16
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt16 "Expected sitkVectorUInt16"
    
    testCase "ToSimpleITK VectorInt16" <| fun _ ->
        let value = PixelType.VectorInt16
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt16 "Expected sitkVectorInt16"
    
    testCase "ToSimpleITK VectorUInt32" <| fun _ ->
        let value = PixelType.VectorUInt32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt32 "Expected sitkVectorUInt32"
    
    testCase "ToSimpleITK VectorInt32" <| fun _ ->
        let value = PixelType.VectorInt32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt32 "Expected sitkVectorInt32"
    
    testCase "ToSimpleITK VectorUInt64" <| fun _ ->
        let value = PixelType.VectorUInt64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorUInt64 "Expected sitkVectorUInt64"
    
    testCase "ToSimpleITK VectorInt64" <| fun _ ->
        let value = PixelType.VectorInt64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorInt64 "Expected sitkVectorInt64"
    
    testCase "ToSimpleITK VectorFloat32" <| fun _ ->
        let value = PixelType.VectorFloat32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat32 "Expected sitkVectorFloat32"
    
    testCase "ToSimpleITK VectorFloat64" <| fun _ ->
        let value = PixelType.VectorFloat64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkVectorFloat64 "Expected sitkVectorFloat64"
    
    testCase "ToSimpleITK LabelUInt8" <| fun _ ->
        let value = PixelType.LabelUInt8
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkLabelUInt8 "Expected sitkLabelUInt8"
    
    testCase "ToSimpleITK LabelUInt16" <| fun _ ->
        let value = PixelType.LabelUInt16
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkLabelUInt16 "Expected sitkLabelUInt16"
    
    testCase "ToSimpleITK LabelUInt32" <| fun _ ->
        let value = PixelType.LabelUInt32
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkLabelUInt32 "Expected sitkLabelUInt32"
    
    testCase "ToSimpleITK LabelUInt64" <| fun _ ->
        let value = PixelType.LabelUInt64
        let result = value.ToSimpleITK()
        Expect.equal result itk.simple.PixelIDValueEnum.sitkLabelUInt64 "Expected sitkLabelUInt64"
    
    testCase "Zero value for UInt8" <| fun _ ->
        let value = PixelType.UInt8
        let result = value.Zero()
        Expect.equal result (box 0uy) "Expected boxed 0uy"
    
    testCase "Zero value for Int8" <| fun _ ->
        let value = PixelType.Int8
        let result = value.Zero()
        Expect.equal result (box 0y) "Expected boxed 0y"
    
    testCase "Zero value for UInt16" <| fun _ ->
        let value = PixelType.UInt16
        let result = value.Zero()
        Expect.equal result (box 0us) "Expected boxed 0us"
    
    testCase "Zero value for Int16" <| fun _ ->
        let value = PixelType.Int16
        let result = value.Zero()
        Expect.equal result (box 0s) "Expected boxed 0s"
    
    testCase "Zero value for UInt32" <| fun _ ->
        let value = PixelType.UInt32
        let result = value.Zero()
        Expect.equal result (box 0u) "Expected boxed 0u"
    
    testCase "Zero value for Int32" <| fun _ ->
        let value = PixelType.Int32
        let result = value.Zero()
        Expect.equal result (box 0) "Expected boxed 0"
    
    testCase "Zero value for UInt64" <| fun _ ->
        let value = PixelType.UInt64
        let result = value.Zero()
        Expect.equal result (box 0UL) "Expected boxed 0UL"
    
    testCase "Zero value for Int64" <| fun _ ->
        let value = PixelType.Int64
        let result = value.Zero()
        Expect.equal result (box 0L) "Expected boxed 0L"
    
    testCase "Zero value for Float32" <| fun _ ->
        let value = PixelType.Float32
        let result = value.Zero()
        Expect.equal result (box 0.0f) "Expected boxed 0.0f"
    
    testCase "Zero value for Float64" <| fun _ ->
        let value = PixelType.Float64
        let result = value.Zero()
        Expect.equal result (box 0.0) "Expected boxed 0.0"
    
    testCase "Zero value for ComplexFloat32" <| fun _ ->
        let value = PixelType.ComplexFloat32
        let result = value.Zero()
        Expect.equal result (box 0.0f) "Expected boxed 0.0f"
    
    testCase "Zero value for ComplexFloat64" <| fun _ ->
        let value = PixelType.ComplexFloat64
        let result = value.Zero()
        Expect.equal result (box 0.0) "Expected boxed 0.0"
    
    testCase "Zero value for VectorUInt8" <| fun _ ->
        let value = PixelType.VectorUInt8
        let result = value.Zero()
        Expect.equal result (box 0uy) "Expected boxed 0uy"
    
    testCase "Zero value for VectorInt8" <| fun _ ->
        let value = PixelType.VectorInt8
        let result = value.Zero()
        Expect.equal result (box 0y) "Expected boxed 0y"
    
    testCase "Zero value for VectorUInt16" <| fun _ ->
        let value = PixelType.VectorUInt16
        let result = value.Zero()
        Expect.equal result (box 0us) "Expected boxed 0us"
    
    testCase "Zero value for VectorInt16" <| fun _ ->
        let value = PixelType.VectorInt16
        let result = value.Zero()
        Expect.equal result (box 0s) "Expected boxed 0s"
    
    testCase "Zero value for VectorUInt32" <| fun _ ->
        let value = PixelType.VectorUInt32
        let result = value.Zero()
        Expect.equal result (box 0u) "Expected boxed 0u"
    
    testCase "Zero value for VectorInt32" <| fun _ ->
        let value = PixelType.VectorInt32
        let result = value.Zero()
        Expect.equal result (box 0) "Expected boxed 0"
    
    testCase "Zero value for VectorUInt64" <| fun _ ->
        let value = PixelType.VectorUInt64
        let result = value.Zero()
        Expect.equal result (box 0UL) "Expected boxed 0UL"
    
    testCase "Zero value for VectorInt64" <| fun _ ->
        let value = PixelType.VectorInt64
        let result = value.Zero()
        Expect.equal result (box 0L) "Expected boxed 0L"
    
    testCase "Zero value for VectorFloat32" <| fun _ ->
        let value = PixelType.VectorFloat32
        let result = value.Zero()
        Expect.equal result (box 0.0f) "Expected boxed 0.0f"
    
    testCase "Zero value for VectorFloat64" <| fun _ ->
        let value = PixelType.VectorFloat64
        let result = value.Zero()
        Expect.equal result (box 0.0) "Expected boxed 0.0"
    
    testCase "Zero value for LabelUInt8" <| fun _ ->
        let value = PixelType.LabelUInt8
        let result = value.Zero()
        Expect.equal result (box 0uy) "Expected boxed 0uy"
    
    testCase "Zero value for LabelUInt16" <| fun _ ->
        let value = PixelType.LabelUInt16
        let result = value.Zero()
        Expect.equal result (box 0us) "Expected boxed 0us"
    
    testCase "Zero value for LabelUInt32" <| fun _ ->
        let value = PixelType.LabelUInt32
        let result = value.Zero()
        Expect.equal result (box 0u) "Expected boxed 0u"
    
    testCase "Zero value for LabelUInt64" <| fun _ ->
        let value = PixelType.LabelUInt64
        let result = value.Zero()
        Expect.equal result (box 0UL) "Expected boxed 0UL"
    
  ]

[<EntryPoint>]
let main argv =
  runTestsWithArgs defaultConfig argv pixelTypeTests
