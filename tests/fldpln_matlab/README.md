# The original FLDPLN model

`fldpln_model_v5ram.m` and `readbilheader.m` are the University of Kansas's MATLAB FLDPLN model, by Jude Kastens,
copied unchanged from [AlabamaWaterInstitute/fldpln](https://github.com/AlabamaWaterInstitute/fldpln) (`src/`) under
the MIT license in [LICENSE](LICENSE).

`run_segments.m` runs the model on the segments `tests/test_fldpln_solvers.py` writes for it, spilling to a steady
state and once a step, so that the test can check that the Python port, `tests/fldpln_model_v5ram.py`, gives the same
tables. The test runs only where MATLAB is on the `PATH`.

The model expects every cell on the raster's edge to point off the raster: a spill that runs onto an edge cell
without a flow direction (0, as 207 edge cells of the test's flow directions are) stops it with an error, so the test
leaves out the segments whose spills do.
