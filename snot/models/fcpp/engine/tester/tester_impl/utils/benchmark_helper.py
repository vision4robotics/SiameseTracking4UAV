# -*- coding: utf-8 -*
import numpy as np

from snot.models.fcpp.pipeline.pipeline_base import PipelineBase


class PipelineTracker(object):
    def __init__(self,
                 name: str,
                 pipeline: PipelineBase,
                 is_deterministic: bool = True):
        """Helper tracker for comptability with 
        
        Parameters
        ----------
        name : str
            [description]
        pipeline : PipelineBase
            [description]
        is_deterministic : bool, optional
            [description], by default False
        """
        self.name = name
        self.is_deterministic = is_deterministic
        self.pipeline = pipeline

    def init(self, image: np.array, box):
        """Initialize pipeline tracker
        
        Parameters
        ----------
        image : np.array
            image of the first frame
        box : np.array or List
            tracking bbox on the first frame
            formate: (x, y, w, h)
        """
        self.pipeline.init(image, box)

    def update(self, image: np.array):
        """Perform tracking
        
        Parameters
        ----------
        image : np.array
            image of the current frame
        
        Returns
        -------
        np.array
            tracking bbox
            formate: (x, y, w, h)
        """
        return self.pipeline.update(image)

