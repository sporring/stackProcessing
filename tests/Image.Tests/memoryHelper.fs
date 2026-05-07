module Tests.memoryHelper

open Expecto
open itk.simple
open Image

[<Tests>]
let memoryHelperTests =
  testList "Memory helper tests" [

    testCase "memoryEstimate for uint8 2x2 image with 1 component" <| fun _ ->
      let arr = array2D [ [1uy; 2uy]; [3uy; 4uy] ]
      let img = Image<uint8>.ofArray2D arr
      let expectedSize = uint32 (2 * 2 * 1 * sizeof<uint8>)
      Expect.equal (Image.memoryEstimateSItk img.Image) expectedSize "Memory estimate for 2x2 uint8"

    testCase "ImageFacts reports size, voxel count, and memory bytes" <| fun _ ->
      let img = Image<uint16>.ofArray2D (array2D [ [1us; 2us; 3us]; [4us; 5us; 6us] ])
      let facts = img.GetFacts()
      Expect.equal facts.Backend "SimpleITK" "Facts should identify the concrete backend."
      Expect.equal facts.Size [2UL; 3UL] "Facts should report the SimpleITK image size."
      Expect.equal facts.VoxelCount 6UL "Facts should count pixels/voxels."
      Expect.equal facts.ComponentBytes 2UL "UInt16 should use two bytes per component."
      Expect.equal facts.ComponentsPerPixel 1UL "Scalar UInt16 should have one component per pixel."
      Expect.equal facts.MemoryBytes 12UL "Memory bytes should be voxel count times component bytes."

    testCase "ImageFacts accounts for vector component count" <| fun _ ->
      let arr = Array3D.init 2 3 4 (fun x y k -> x + y + k)
      let img = Image<int list>.ofArray3DVector arr
      let facts = img.GetFacts()
      Expect.equal facts.Size [2UL; 3UL] "Vector images should report spatial dimensions, not the component axis."
      Expect.equal facts.VoxelCount 6UL "Voxel count should only include spatial positions."
      Expect.equal facts.ComponentsPerPixel 4UL "Vector component count should be part of the memory facts."
      Expect.equal facts.MemoryBytes (6UL * 4UL * facts.ComponentBytes) "Memory bytes should include every component per pixel."

    testCase "ImageFacts can estimate typed image memory without an allocated image" <| fun _ ->
      Expect.equal (ImageFacts.memoryBytesForType<float> 6UL 2u) 96UL "Six two-component float64 pixels should use 96 bytes."
      Expect.equal (ImageFacts.sliceBytesForType<uint16> 4u 5u) 40UL "A 4x5 UInt16 slice should use 40 bytes."
  ]
  
