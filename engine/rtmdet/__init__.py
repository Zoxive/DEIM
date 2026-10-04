"""RTMDet, ported from MMDetection 3.3.0 (Apache-2.0) with no OpenMMLab dependency; see model.py."""

from .model import RTMDet
from .criterion import RTMDetCriterion
from .postprocessor import RTMDetPostProcessor
from .distill import DEIMTeacher
