# DMTPRL 
## Motivation:
The geometric consistency of AlphaFold2 is maintained on the high-dimensional edge representations.

Geoformer could optimize high-dimensional representations to capture complex interaction patterns among amino acids while still maintaining geometric consistency in Euclidean space. In each Geoformer layer, these embeddings are updated iteratively to refine the geometric inconsistency: each node embedding was updated with related pairwise embeddings, and each pairwise embedding was updated by triangular consistency of pairwise embeddings.

However, most protein methods learned protein representations are usually not constrained, leading to performance degradation due to data scarcity, task adaptation, etc. Can we design a loss to satisfy the demand for geometric consistency?



## Main codes
for 
[Deep Manifold Graph Auto-Encoder For Attributed Graph Embedding](https://ieeexplore.ieee.org/abstract/document/10095904) 
and
[Deep Manifold Transformation for Protein Representation Learning](https://arxiv.org/abs/2402.09416) 

Codes are based on DMT and KeAP. 

KeAP: [paper](https://openreview.net/forum?id=VbCMhg7MRmj) and [codes](https://github.com/RL4M/KeAP) 

DMT: [paper](https://arxiv.org/abs/2207.03160) and [codes](https://github.com/zangzelin/code_ECCV2022_DLME) 


