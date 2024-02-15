import numpy as np
from scipy.stats import wasserstein_distance

def EMD(a, b):
  return wasserstein_distance(np.arange(len(a)), np.arange(len(b)), a,b)

def earth_mover_distance(dirt, holes):

  dirt_c = np.copy(dirt) 
  holes_c = np.copy(holes)
  tot_work = 0.0

  while True:  
    from_idx = first_nonzero(dirt_c)
    to_idx = first_nonzero(holes_c)
    if from_idx == -1 or to_idx == -1:
      break
    (flow, dist, dirt_c, holes_c) = \
      move_dirt(dirt_c, from_idx, holes_c, to_idx)
    tot_work += flow * dist
  return tot_work

def move_dirt(dirt, from_idx, holes, to_idx):
  if dirt[from_idx] <= holes[to_idx]:  
    flow = dirt[from_idx]

    dirt[from_idx] = 0.0  # all dirt got moved
    holes[to_idx] -= flow  # less to fill now
  elif dirt[from_idx] > holes[to_idx]:  # use just part of dirt
    flow = holes[to_idx]  # fill remainder of hole
 
    dirt[from_idx] -= flow
    holes[to_idx] = 0.0  # hole is filled
  dist = np.abs(from_idx - to_idx)
  return flow, dist, dirt, holes

def first_nonzero(dirt_or_holes):
  dim = len(dirt_or_holes)
  for i in range(dim):
    if dirt_or_holes[i] > 0.0:  
      return i
  return -1  # no cells found