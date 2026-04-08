# kmeanscluster

A from-scratch K-Means clustering implementation in Python using soft (fuzzy) assignments.

## Overview

This project demonstrates unsupervised machine learning via K-Means clustering. Rather than hard cluster assignments, it uses a soft assignment approach with a temperature parameter (`beta`) that controls how sharply points are assigned to cluster centers. The algorithm iterates between updating assignments and recomputing means, stopping early when the cost change falls below a threshold.

The script generates synthetic 2-D data from three Gaussian distributions, runs the clustering, and plots the result at each iteration using a subplot grid.

## How to Run

**1. Install dependencies**

```bash
pip install numpy matplotlib
```

**2. Run the script**

```bash
python kmeans.py
```

The `main()` function generates 900 sample points across 3 clusters and runs the algorithm with K=3. Cluster separation and the number of iterations can be adjusted by editing the `s` and `max_iter` variables directly in the script.

## Algorithm Details

- Soft assignment: each point's membership `R[n, k]` is computed via a softmax over distances, scaled by `beta`
- Centroid update: means are recomputed as the weighted average of all points under each cluster
- Convergence: stops when `|cost[i] - cost[i-1]| < 1e-4` or `max_iter` is reached
- Distance metric: squared Euclidean distance

## Dependencies

- numpy
- matplotlib
