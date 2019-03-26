import random
from typing import Tuple, Type

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


class Sequence(keras.utils.Sequence):
    batches_per_epoch = None

    def __init__(
            self,
            batch_size: int,
            features,
            responses,
            weights,
            feature_scaler: Type[BaseGlobalScaler],
            response_scaler: Type[BaseGlobalScaler],
            apply_random_transformations: bool = False
    ) -> None:
        self.features = features
        self.responses = responses
        self.weights = weights
        self.batch_size = batch_size
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.apply_random_transformations = apply_random_transformations
        self.current_fold = 0

        self.total_samples = 0
        self.sample_index = 0
        for fold in range(len(features)):
            self.total_samples += features[fold].shape[0]

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, item: int, apply_transforms: bool = True) -> Tuple[np.array, np.array]:
        # TODO:  I think this should work because apply_transforms is a kwarg
        # Method is required for Keras functionality
        _logger.debug('Get batch {} with {} items via sequence'.format(item, self.batch_size))

        batch_features = (self.features[self.current_fold])[self.sample_index:self.batch_size, ...]
        batch_responses = (self.responses[self.current_fold])[self.sample_index:self.batch_size, ...]
        batch_weights = (self.weights[self.current_fold])[self.sample_index:self.batch_size, ...]

        breakout = False
        while (batch_features.shape[0] != self.batch_size):
            self.sample_index = 0
            self.current_fold += 1

            if (self.current_fold == len(self.features)):
                breakout = True
                break

            batch_features = np.append(batch_features,(self.features[self.current_fold])[self.sample_index:self.batch_size, ...], axis=0)
            batch_responses = np.append(batch_responses,(self.responses[self.current_fold])[self.sample_index:self.batch_size, ...], axis=0)
            batch_weights = np.append(batch_weights,(self.weights[self.current_fold])[self.sample_index:self.batch_size, ...], axis=0)

        if (breakout):
            self.current_fold = 0
            self.sample_index = 0
        else:
            self.sample_index += self.batch_size

        batch_features, batch_responses = self._transform_features_and_responses(batch_features, batch_responses)
        batch_responses = np.append(batch_responses,batch_weights,axis=-1)

        return batch_features, batch_responses

    def get_transformed_batch(self, idx_batch):
        return self.__getitem__(idx_batch, apply_transforms=True)

    def get_untransformed_batch(self, idx_batch):
        return self.__getitem__(idx_batch, apply_transforms=False)

    def _transform_features_and_responses(self, features: np.array, responses: np.array) -> Tuple[np.array, np.array]:
        # Required transforms for data quality
        if (self.feature_scaler is not None):
            features = self.feature_scaler.transform(features)
        if (self.response_scaler is not None):
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
