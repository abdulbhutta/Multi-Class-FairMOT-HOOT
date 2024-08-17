from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset.jde import JointDataset, MultiScaleJD


def get_dataset(dataset, task, mult_scale):
  if task == 'hoot':
    if mult_scale:
      return MultiScaleJD
    else:
      return JointDataset
  else:
    return None
