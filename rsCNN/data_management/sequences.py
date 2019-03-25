import random
from typing import Tuple, Type

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


BATCHES_PER_EPOCH = 1


class Sequence(keras.utils.Sequence):
    # TODO:  add functionality to get transformed and untransformed data from a batch
    batches_per_epoch = None

    def __init__(
            self,
            batch_size: int,
            feature_scaler: Type[BaseGlobalScaler],
            response_scaler: Type[BaseGlobalScaler],
            apply_random_transformations: bool = False
    ) -> None:
        self.batch_size = batch_size
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.apply_random_transformations = apply_random_transformations
        super().__init__(BATCHES_PER_EPOCH)

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        return BATCHES_PER_EPOCH

    def __getitem__(self, item: int, apply_transforms: bool = True) -> Tuple[np.array, np.array]:
        # TODO:  I think this should work because apply_transforms is a kwarg
        # Method is required for Keras functionality
        _logger.debug('Get batch {} with {} items via sequence'.format(item, self.batch_size))
        batch_features = list()
        batch_responses = list()
        for idx_pair in range(self.batch_size):
            features, responses = self._get_sample('TODO')
            if apply_transforms:
                features, responses = self._transform_features_and_responses(features, responses)
            batch_features.append(features)
            batch_responses.append(responses)
        _logger.debug('Retrieved full batch'.format(self.batch_size))
        # Need list of np arrays for *_generator model methods
        batch_features = [np.array(batch_features)]
        batch_responses = [np.array(batch_responses)]
        return batch_features, batch_responses

    def get_transformed_batch(self, idx_batch):
        return self.__getitem__(idx_batch, apply_transforms=True)

    def get_untransformed_batch(self, idx_batch):
        return self.__getitem__(idx_batch, apply_transforms=False)

    def _get_sample(self) -> Tuple[np.array, np.array]:
        # TODO:  this needs the logic for grabbing a single sample from the built dataset. is this deterministic or
        #  stochastic? if stochastic, do we want to sample with or without replacement? This requires that we pass the
        #  sequence one last piece of information in __init__, which is where the data is located.
        raise NotImplementedError
        return features, responses

    def _transform_features_and_responses(self, features: np.array, responses: np.array) -> Tuple[np.array, np.array]:
        # Required transforms for data quality
        features = self.feature_scaler.transform(features)
        responses = self.response_scaler.transform(responses)
        # Return without transformations if testing and not training
        if not self.apply_random_transformations:
            return features, responses
        # Optional transforms for training variety
        # Flip top to bottom
        if random.random() > 0.5:
            features = np.flip(features, axis=0)
            responses = np.flip(responses, axis=0)
        # Flip side to side
        if random.random() > 0.5:
            features = np.flip(features, axis=1)
            responses = np.flip(responses, axis=1)
        # Rotate 0, 1, 2, or 3 times
        num_rotations = np.floor(4 * random.random())
        features = np.rot90(features, k=num_rotations, axes=(0, 1))
        responses = np.rot90(responses, k=num_rotations, axes=(0, 1))
        return features, responses
