# The generic 3DGN framework

This is the 3D graph network (3DGN) proposed in [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2).

<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/dig/3dgraph/3dgn/figs/frame.png" width="600" class="center" alt="3dgn"/>
    <br/>
</p>


## Functions need implemented: three ɸ functions and six ρ functions.
### ɸ functions:
update functions, applied to nodes, edges, or the whole graph as information update functions for the corresponding geometries.
### ρ functions:
aggreagation functions, used to aggregate information from one type of geometry to another.

# Spherical Message Passing (SMP)

This is the proposed spherical message passing (SMP) as a novel and speciﬁc scheme for realizing the 3DGN framework in the spherical coordinate system(SCS) in [Spherical Message Passing for 3D Graph Networks](https://arxiv.org/abs/2102.05013v2).

<p align="center">
<img src="https://github.com/divelab/DIG/blob/main/dig/3dgraph/3dgn/figs/frame1.png" width="600" class="center" alt="smp"/>
    <br/>
</p>

## Functions need implemented: three ɸ functions and three ρ functions.
### ɸ functions:
update functions, applied to nodes, edges, or the whole graph as information update functions for the corresponding geometries.
### ρ functions:
aggreagation functions, used to aggregate information from one type of geometry to another.
