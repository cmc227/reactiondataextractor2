# -*- coding: utf-8 -*-
"""
Recognise
========

This module contains optical chemical structure recognition tools and routines.

author: Damian Wilary
email: dmw51@cam.ac.uk

Recognition is achieved using OSRA and performed via a pyOsra wrapper.
"""
import os
import itertools
import logging
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn

from DECIMER.config import get_bnw_image, delete_empty_borders, central_square_image, PIL_im_to_BytesIO, get_resize, increase_contrast
from DECIMER.decimer import tokenizer, DECIMER_V2

from models.reaction import Diagram
from reactiondataextractor.models.segments import FigureRoleEnum, Figure
from utils.utils import isolate_patches

log = logging.getLogger()


class DecimerRecogniser:
    def __init__(self, model_id='Canonical'):
        assert model_id.capitalize() in ['Canonical', 'Isomeric', 'Augmented'], "model_id has to be one of the following:\
                                                                            ['Canonical', 'Isomeric', 'Augmented']"
        self.model = DECIMER_V2

    def decode_image(self, img: np.ndarray) -> 'Tensor':
        """
        Loads and preprocesses an image
        :param img: image array for preprocessing
        :type img: np.ndarray
        """
        # img = self.remove_transparent(img)
        img = increase_contrast(img)
        img = get_bnw_image(img)
        img = get_resize(img)
        img = delete_empty_borders(img)
        img = central_square_image(img)
        img = PIL_im_to_BytesIO(img)
        img = tf.image.decode_png(img.getvalue(), channels=3)
        img = tf.image.resize(img, (512, 512), method="gaussian", antialias=True)
        img = efn.preprocess_input(img)
        return img

    def detokenize_output(self, predicted_array: 'Tensor') -> str:
        """
        This function takes the predited tokens from the DECIMER model
        and returns the decoded SMILES string.
        :param predicted_array: Tensor of shape [1, n_tokens] of predicted tokens from DECIMER
        :type predicted_array: Tensor
        :return: smiles representation of a diagram
        :rtype: str"""
        # Check if predicted_array is a tuple and handle accordingly
        if isinstance(predicted_array, tuple):
            # Unpack the tuple if needed; assuming the first element is the relevant tensor
            predicted_array = predicted_array[0]  # Use the first element in the tuple

        # Ensure predicted_array is a tensor and check its dtype
        if isinstance(predicted_array, tf.Tensor):
            if predicted_array.dtype != tf.int32:
                predicted_array = tf.cast(predicted_array, tf.int32)  # Cast to int32 if not already

        # Squeeze the tensor to remove dimensions of size 1, then convert to numpy
        outputs = [tokenizer.index_word[i] for i in tf.squeeze(predicted_array).numpy()]

        # Construct the SMILES string, removing <start> and <end> tokens
        prediction = (
            "".join([str(elem) for elem in outputs])
            .replace("<start>", "")
            .replace("<end>", "")
        )

        return prediction

    

        return prediction
    
