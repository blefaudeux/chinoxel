# chinoxel

Trying to implement [Plenoxels](https://alexyu.net/plenoxels/) using [Taichi](https://docs.taichi.graphics/). Reimplementing from scratch as a comprehension exercise, not looking at the original code. First project with Taichi so a lot of parts may not be very idiomatic..


TODOs:

- [x] General framework (grid/render/compare with views/optimize)
- [x] Handle camera extrinsics
- [x] Interpolate node contributions
- [x] Color buffers
- [x] Propagate rays over depth
- [ ] Manual backward pass implementation (store gradients). 
  Taichi's autodiff dies on the color interpolation scheme, and on the interestection, which is sparse by nature. May be a better idea to compute the gradients by hand, and store them in a dedicated buffer as the rendering is being done. One possible issue is that the same node can contribute to multiple pixels. Their gradient would differ, and would then need to be traced back to the right node contribution. A naive take is to consider [pixels]x[nodes] gradients, but this would be very sparse and waste tons of memory. Maybe that per-node gradient storage (per pixel) is the better take, at most 4 pixels would be stored there. 

- [ ] Add some unit tests
- [ ] Handle spherical harmonics in the nodes
- [ ] General speedup

<p align="center">
  <img src="orbit.gif" width=480>
</p>
