module Vector

/// <summary>
/// Represents a fixed-length, generic numeric vector.
/// </summary>
/// <typeparam name="'T">
/// The numeric type of the vector elements. Must support the addition operator <c>(+)</c>.
/// </typeparam>
type Vector<'T> = Vector of 'T[]

/// <summary>
/// Returns an array of index positions corresponding to the elements in a vector.
/// </summary>
/// <param name="vector">The <c>Vector&lt;'T&gt;</c> whose indices will be returned.</param>
/// <returns>
/// An array of integers representing the zero-based indices of the vector elements.
/// </returns>
/// <remarks>
/// This function is useful when generating axis labels or bin positions for plotting or iteration.
/// The result is always an array <c>[|0; 1; ...; n - 1|]</c> where <c>n</c> is the length of the vector.
/// </remarks>
/// <example>
/// <code>
/// let v = Vector.ofArray [| 10; 20; 30 |]
/// let idx = Vector.indices v  // returns [| 0; 1; 2 |]
/// </code>
/// </example>
let indices (Vector v) = 
    Array.mapi (fun i _ -> i) v

/// <summary>
/// Creates a zero-initialized vector of the given length using the provided zero value.
/// </summary>
/// <param name="zeroValue">The value used to initialize all elements (typically 0, 0.0, etc.).</param>
/// <param name="length">The desired length of the vector.</param>
/// <returns>A new vector of type <c>Vector&lt;'T&gt;</c> where each element is <c>zeroValue</c>.</returns>
let zero (zeroValue: 'T) (length: int) : Vector<'T> =
    Vector (Array.init length (fun _ -> zeroValue))

/// <summary>
/// Adds two vectors of the same length element-wise using the <c>(+)</c> operator.
/// </summary>
/// <param name="v1">The first vector operand.</param>
/// <param name="v2">The second vector operand.</param>
/// <returns>A new vector where each element is the sum of corresponding elements from <c>v1</c> and <c>v2</c>.</returns>
/// <exception cref="System.ArgumentException">
/// Thrown if the input vectors are of different lengths.
/// </exception>
let inline add (Vector v1: Vector<'T>) (Vector v2: Vector<'T>) : Vector<'T>
    when 'T: (static member (+) : 'T * 'T -> 'T) =
    if v1.Length <> v2.Length then
        invalidArg "v2" "Vectors must be of the same length to add."
    Array.map2 (+) v1 v2 |> Vector

/// <summary>
/// Creates a vector from a given array by copying its contents.
/// </summary>
/// <param name="arr">The input array.</param>
/// <returns>A new <c>Vector&lt;'T&gt;</c> that wraps a copy of the input array.</returns>
let ofArray (arr: 'T[]) : Vector<'T> =
    Vector (Array.copy arr)

/// <summary>
/// Extracts the underlying array from a vector.
/// </summary>
/// <param name="v">The vector to extract the array from.</param>
/// <returns>The internal array of type <c>'T[]</c>.</returns>
let toArray (Vector v) = v
