#r "bin/Debug/net8.0/ImageClass.dll"            
#r "bin/Debug/net8.0/SimpleITKCSharpManaged.dll"
open ImageClass
let I = Image<int>([10u;12u],0)
let J = Image<int list>([10u;12u])
printfn "%A" I[1,2]
I[1,2] <- 1
printfn "%A" I[1,2]
printfn "%A" J[2,1]
J[1,2] <- [2;1]
printfn "%A" J[1,2];;
