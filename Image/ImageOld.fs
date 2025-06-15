module Image
open itk.simple

type Image(img: itk.simple.Image) =

    /// Underlying itk.simple image
    member this.Image = img

    /// String representation
    override this.ToString() = img.ToString()

    // ----- Operator Overloads -----

    /// Image + Image
    static member (+) (a: Image, b: Image) : Image =
        let filter = new AddImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    /// Image + scalar
    static member (+) (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new AddImageFilter()
        Image(filter.Execute(a.Image, value))

    /// scalar + Image
    static member (+) (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a + value

    /// Image - Image
    static member (-) (a: Image, b: Image) : Image =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    /// Image - scalar
    static member (-) (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(a.Image, value))

    /// scalar - Image
    static member (-) (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(value, a.Image))

    /// Image * Image
    static member (*) (a: Image, b: Image) : Image =
        let filter = new MultiplyImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    /// Image * scalar
    static member (*) (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new MultiplyImageFilter()
        Image(filter.Execute(a.Image, value))

    /// scalar * Image
    static member (*) (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a * value

    /// Image / Image
    static member (/) (a: Image, b: Image) : Image =
        let filter = new DivideImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    /// Image / scalar
    static member (/) (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new DivideImageFilter()
        Image(filter.Execute(a.Image, value))

    /// scalar / Image
    static member (/) (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new DivideImageFilter()
        Image(filter.Execute(value, a.Image))

    // operator overloading in F# is not easy. 
    //   (=) etc. cannot be overloaded directly. 
    //   op_Equality etc. can but must return bools so cannot return Image of bools and cannot be curried directly
    //   when ^T : (static member op_Explicit : ^T -> double) can be used for type checking
    // Thus we settle for boxing:
    //   + Single function handles all needed combinations.
    //   + Allows generic numeric types: int, float, byte, etc.
    //   + Works naturally in F# functional code (List.map ((+) 5.0)).
    //   + Avoids F#'s restriction against multiple function/operator overloads.
    //   - Type resolution for Image vs scalar happens at runtime via box, not static dispatch.
    //   - You lose type-checking for "is this a scalar?" - but retain type-checking for Image.

    // Equality: Image = Image
    static member op_Equality (a: Image, b: Image) : Image =
        let filter = new EqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    // Equality: Image = scalar
    static member op_Equality (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new EqualImageFilter()
        Image(filter.Execute(a.Image, value))

    // Equality: scalar = Image
    static member op_Equality (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a = value

    // Inequality: <>
    static member op_Inequality (a: Image, b: Image) : Image =
        let filter = new NotEqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    static member op_Inequality (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new NotEqualImageFilter()
        Image(filter.Execute(a.Image, value))

    static member op_Inequality (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a <> value

    // Less than: <
    static member op_LessThan (a: Image, b: Image) : Image =
        let filter = new LessImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    static member op_LessThan (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessImageFilter()
        Image(filter.Execute(a.Image, value))

    static member op_LessThan (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(value, a.Image))

    // Less than or equal: <=
    static member op_LessThanOrEqual (a: Image, b: Image) : Image =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    static member op_LessThanOrEqual (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(a.Image, value))

    static member op_LessThanOrEqual (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(value, a.Image))

    // Greater than: >
    static member op_GreaterThan (a: Image, b: Image) : Image =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    static member op_GreaterThan (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(a.Image, value))

    static member op_GreaterThan (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessImageFilter()
        Image(filter.Execute(value, a.Image))

    // Greater than or equal: >=
    static member op_GreaterThanOrEqual (a: Image, b: Image) : Image =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    static member op_GreaterThanOrEqual (a: Image, value: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(a.Image, value))

    static member op_GreaterThanOrEqual (value: ^T, a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(value, a.Image))
