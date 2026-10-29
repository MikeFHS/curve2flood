# Kernel-Weighted Mapper

`mapper: Curve2Flood-Kernel Weighted`, the default. Each stream cell spreads its water surface over the cells around
it, and the cells where the spread water surface is above the ground flood. It is fast and needs nothing beyond the
rating curves.

## How it works

Each stream cell $s$ has a water surface elevation $\text{WSE}_s$ and a top width $T_s$ from its rating curve. Its
water reaches every cell up to one top width away, in a square window $n_s = T_s / \Delta x$ cells each way, where
$\Delta x$ is the cell size. A cell $x$ in the window gets the weight

$$
w(s, x) = \frac{1}{d(s, x)^2}
$$

where $d$ is the distance between the two cells' centres in metres. The stream cell itself gets $10^4$, as if it were
1 cm away. Where windows overlap, the water surface is the weighted average of every stream cell that reaches the
cell:

$$
\text{WSE}(x) = \frac{\sum_s w(s, x) \, \text{WSE}_s}{\sum_s w(s, x)}
$$

and the cell floods where $\text{WSE}(x)$ is above its elevation. So near a stream cell its own water surface
dominates, and between two stream cells the surface blends from one to the other.

![Left: a cross section of a valley; one stream cell's water surface floods the cells below it within one top width each way, but not a dip beyond. Right: two stream cells' water surfaces, 103 m and 101 m, blend between them.](figures/kernel_weighted.svg)

Details:

- A top width is the rating curve's times `TW_MultFact`, capped at the widest reach's median top width, and at
  `TopWidthPlausibleLimit`. 
- Stream cells without a top width, or whose water surface is less than 1 mm above the DEM, don't spread.
- With `LocalFloodOption`, a stream cell's water reaches only the cells of its window that are below its water
  surface and connect to it, so it can't jump over a ridge into the next valley. The connection is traced in a few
  sweeps across the window, so a winding path can be missed.
- The stream cells themselves are always flooded.
- With `OutVEL`, the stream cells' slopes are averaged with the same weights: their valley slopes outside the
  channels, when the VDT database has them (see [Outputs](outputs.md#velocity)).

## Set depth

With `Set_Depth`, every stream cell of `Stream_File` gets a water surface `Set_Depth` metres above the DEM and a top
width of `TopWidthPlausibleLimit`. With a wide limit and many stream cells this can be slow, since every stream cell
spreads over the whole window.
