# chinoxel

Trying to implement [Plenoxels](https://alexyu.net/plenoxels/) using [Taichi](https://docs.taichi.graphics/). Reimplementing from scratch as a comprehension exercise, not looking at the original code. First project with Taichi so a lot of parts may not be very idiomatic.. 


TODOs:

- [x] General framework (grid/render/compare with views/optimize)
- [ ] Handle camera extrinsics
- [ ] Manual backward pass implementation (store gradients)
- [ ] Add some unit tests
- [ ] Handle spherical harmonics in the nodes
- [x] Interpolate node contributions
- [x] Color buffers
- [ ] General speedup
- [x] Propagate rays over depth

<p align="center">
  <img src="chinoxel.png" width=600>
</p>
