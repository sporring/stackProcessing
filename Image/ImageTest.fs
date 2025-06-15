
module Image

open itk.simple
open Microsoft.FSharp.Core.LanguagePrimitives

type Image(img: itk.simple.Image) =

    /// Underlying itk.simple Image
    member this.Image = img

    override this.ToString() = img.ToString()

    /// Static constructor from SimpleITK Image
    static member FromSimpleITK(img: itk.simple.Image) = Image(img)


/// Module with inline operator overloads for Image
[<AutoOpen>]
module ImageOps =

    let inline (+) (a: Image) (b: Image) : Image =
        let filter = new AddImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (+) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new AddImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (+) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a + v

    let inline (-) (a: Image) (b: Image) : Image =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (-) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (-) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new SubtractImageFilter()
        Image(filter.Execute(double v, a.Image))

    let inline (*) (a: Image) (b: Image) : Image =
        let filter = new MultiplyImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (*) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new MultiplyImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (*) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        a * v

    let inline (/) (a: Image) (b: Image) : Image =
        let filter = new DivideImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (/) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new DivideImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (/) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new DivideImageFilter()
        Image(filter.Execute(double v, a.Image))

    let inline (>=) (a: Image) (b: Image) : Image =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (>=) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (>=) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterEqualImageFilter()
        Image(filter.Execute(double v, a.Image))

    let inline (<=) (a: Image) (b: Image) : Image =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (<=) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (<=) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessEqualImageFilter()
        Image(filter.Execute(double v, a.Image))

    let inline (>) (a: Image) (b: Image) : Image =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (>) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (>) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new GreaterImageFilter()
        Image(filter.Execute(double v, a.Image))

    let inline (<) (a: Image) (b: Image) : Image =
        let filter = new LessImageFilter()
        Image(filter.Execute(a.Image, b.Image))

    let inline (<) (a: Image) (v: ^T) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessImageFilter()
        Image(filter.Execute(a.Image, double v))

    let inline (<) (v: ^T) (a: Image) : Image when ^T : (static member op_Explicit : ^T -> double) =
        let filter = new LessImageFilter()
        Image(filter.Execute(double v, a.Image))
