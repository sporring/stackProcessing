namespace CoreTypes

open itk.simple

/// <summary>
/// Represents a slice of a stack of 2d images. 
/// </summary>
type ImageSlice = {
    Index: uint
    Image: Image
}
