
# FSharp.Image Library

> **Namespace:** `FSharp`
> **Module:** `Image`
> **Purpose:** A functional F# wrapper around **SimpleITK** for efficient manipulation of large (possibly larger-than-memory) 3D and multidimensional images with minimal I/O overhead.

---

## Overview

The `Image` module provides a high-level F# interface to the [SimpleITK](https://simpleitk.org) imaging library. It includes utilities for:

* Typed conversions between F# lists and SimpleITK vectors.
* Type-safe creation, manipulation, and arithmetic of `Image<'T>` objects.
* Efficient pixel-level access, memory usage tracking, and safe interop with ITK data structures.
* Handling large image datasets where both **memory** and **I/O bandwidth** are limiting factors.

---

## InternalHelpers Module

> Utilities for bridging between F# types and SimpleITK types.
> Most users won't need to call these directly.

### Vector Conversion

| Function                                                      | Description                                                      |
| ------------------------------------------------------------- | ---------------------------------------------------------------- |
| `toVectorUInt8`, `toVectorInt8`, ..., `toVectorFloat64`       | Convert F# lists to the corresponding `itk.simple.Vector*` type. |
| `fromVectorUInt8`, `fromVectorInt8`, ..., `fromVectorFloat64` | Convert an ITK vector back to an F# list.                        |
| `fromItkVector f v`                                           | Generic vector transformation using function `f`.                |

### Image and Type Helpers

| Function             | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `fromType<'T>`       | Returns the corresponding `itk.simple.PixelIDValueEnum` for type `'T`. |
| `ofCastItk<'T>`      | Cast an existing ITK image to a new pixel type `'T`.                   |
| `array2dZip a b`     | Zip two 2D arrays elementwise.                                         |
| `pixelIdToString id` | Get string representation of a SimpleITK pixel ID.                     |
| `flatIndices size`   | Enumerate all flattened indices for an N-dimensional image size.       |

### Boxed Pixel Access

| Function                        | Description                                         |
| ------------------------------- | --------------------------------------------------- |
| `setBoxedPixel img t idx value` | Set a pixel value using boxed (generic) objects.    |
| `getBoxedPixel img t idx`       | Get a boxed pixel value from an image.              |
| `getBoxedZero t vSize`          | Create a boxed zero value for a pixel type.         |
| `mulAdd t acc k p`              | Multiply and add operation (used for accumulation). |

---

## Global Functions

| Function                     | Description                                              |
| ---------------------------- | -------------------------------------------------------- |
| `getBytesPerComponent t`     | Get the number of bytes per F# component type.           |
| `getBytesPerSItkComponent t` | Get bytes per SimpleITK pixel type.                      |
| `equalOne v`                 | Returns `true` if `v` equals one.                        |
| `printDebugMessage str`      | Print internal debug output (when debugging is enabled). |

---

## Internal State

| Value         | Description                               |
| ------------- | ----------------------------------------- |
| `syncRoot`    | Synchronization object for thread safety. |
| `totalImages` | Mutable counter for total images managed. |
| `memUsed`     | Tracks total allocated image memory.      |
| `debug`       | Global debug flag.                        |

---

## Image<'T> Type

> Represents a typed, possibly large, N-dimensional image.

```fsharp
type Image<'T when 'T: equality> =
    new : sz:uint list
        * ?optionalNumberComponents:uint
        * ?optionalName:string
        * ?optionalIndex:int
        * ?optionalQuiet:bool -> Image<'T>
```

### Interfaces

* `System.IComparable`
* `System.IEquatable<Image<'T>>`

### Example

```fsharp
open FSharp.Image

let img = Image<float>([256u; 256u; 128u], optionalName="BrainMRI")
```

### Properties (inferred from ITK integration)

| Property     | Type            | Description                     |
| ------------ | --------------- | ------------------------------- |
| `Size`       | `uint list`     | Size of each dimension.         |
| `PixelType`  | `'T`            | Pixel data type.                |
| `Name`       | `string option` | Optional image identifier.      |
| `Components` | `uint`          | Number of components per pixel. |

### Operators

| Operator | Signature                            | Description                     |
| -------- | ------------------------------------ | ------------------------------- |
| `(&&&)`  | `Image<'T> * Image<'T> -> Image<'T>` | Logical AND between two images. |
| `(*)`    | `Image<'T> * Image<'T> -> Image<'T>` | Elementwise multiplication.     |
| `(+)`    | `Image<'T> * Image<'T> -> Image<'T>` | Elementwise addition.           |
| `(-)`    | `Image<'T> * Image<'T> -> Image<'T>` | Elementwise subtraction.        |
| `(/)`    | `Image<'T> * Image<'T> -> Image<'T>` | Elementwise division.           |

---

## Usage Example

```fsharp
open FSharp.Image

// Create two large images
let a = Image<float>([512u; 512u; 256u], optionalName="A")
let b = Image<float>([512u; 512u; 256u], optionalName="B")

// Perform pixelwise arithmetic
let result = a + b * 2.0

// Access individual pixels (boxed)
let value = Image.InternalHelpers.getBoxedPixel a a.PixelType (itk.simple.VectorUInt32([|0u; 0u; 0u|]))
printfn "First pixel: %A" value
```

---

## Performance Notes

* The module is optimized for **chunked access** and **lazy evaluation** when possible.
* Avoid copying large images; prefer in-place operations and type-safe conversions.
* Use `debug <- true` to enable detailed memory usage tracking and performance metrics.
