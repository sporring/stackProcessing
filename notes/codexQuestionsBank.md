Questions to consider:

- thresholdZarrChunksUInt8 should be regular threshold. Perhaps Chunks and LocatedChunks should be merged, but we should be careful not to slow everything down, so for now, specialised threshold is ok for the principle.

- I've been thinking about the offset-into thick slice as a representation, where we still let inc/decRef handle memory release of the thick slices, in such a way that the thick slice is inc'ed according to how many thin slices it contains, and whenever a process is done with a thin slice, then the thick slice's counter is dec'ed.

  And rolling windows will then be in terms of the thin slices window overlap with the thick slices. For rolling windows, this will mean that we must store 2 thick slices often, but we will avoid copying into thin slices.

  A thin slice embedded into a thick slice could include the thick slice reference but have its own offset information, and its inc/decRef should point to the thick slice's inc/decs.

  Will this be worth it?

- ChunkSimdOptimizationIdeas.md

- medianFloat32 probably needs versions for other types

- I'm worried about the many uses of unbox and box'ing. If it enters into hot loop, then that'll be costly: https://zetcode.com/fsharp/boxing-unboxing/

---

Ignore anything below here:

- Which functions are we missing wrt. standard microscope and synchrotron image analysis pipelines (Hasse, Euro-BioImaging, Globias, etc.)
