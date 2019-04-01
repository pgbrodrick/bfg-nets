import random
from typing import Tuple

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


class BaseSequence(keras.utils.Sequence):
    batches_per_epoch = None
    feature_scaler = None
    response_scaler = None

    def __init__(self, feature_scaler: BaseGlobalScaler, response_scaler: BaseGlobalScaler):
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        raise NotImplementedError

    # TODO:  Phil:  can we rename apply_transforms to apply_random_transforms for clarity?
    def __getitem__(self, index: int, apply_transforms: bool = True) -> Tuple[np.array, np.array]:
        # Method is required for Keras functionality
        # TODO:  Phil:  I'm trying to make it so we can build our own sequences more easily, even so we can extend the
        #  memmapped sequence to do things like add noise or other modifications. This is a first-pass at that. I think
        #  this general strategy makes it possible because we can more flexibly implement submethods when necessary
        # rather than rewriting __getitem__ every time. Let me know what you think.
        _logger.debug('Get batch {} with {} items via sequence'.format(index, self.batch_size))
        _logger.debug('Get features, responses, and weights')
        features, responses, weights = self._get_features_responses_weights(index)
        _logger.debug('Modify features, responses, and weights prior to scaling')
        features, responses, weights = self._modify_features_responses_weights_before_scaling(
            features, responses, weights)
        _logger.debug('Scale features')
        features = self._scale_features(features)
        _logger.debug('Scale responses')
        responses = self._scale_responses(responses)
        _logger.debug('Append weights to responses for loss functions')
        responses_with_weights = np.append(responses, weights, axis=-1)
        if apply_transforms is True:
            _logger.debug('Apply random transformations to features and responses')
            self._apply_random_transformations(features, responses)

    def get_transformed_batch(self, idx_batch: int):
        # Method is used for visualizations
        return self.__getitem__(idx_batch, apply_transforms=True)

    def get_untransformed_batch(self, idx_batch: int):
        # Method is used for visualizations
        return self.__getitem__(idx_batch, apply_transforms=False)

    def _get_features_responses_weights(self, index: int) -> Tuple[np.array, np.array, np.array]:
        raise NotImplementedError

    def _modify_features_responses_weights_before_scaling(
            self,
            features: np.array,
            responses: np.array,
            weights: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        return features, responses, weights

    def _scale_features(self, features: np.array) -> np.array:
        return self.feature_scaler.transform(features)

    def _scale_responses(self, responses: np.array) -> np.array:
        return self.response_scaler.transform(responses)

    def _apply_random_transformations(self, features: np.array, responses: np.array) -> Tuple[np.array, np.array]:
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


class MemmappedSequence(BaseSequence):
    batches_per_epoch = None

    def __init__(
            self,
            features,
            responses,
            weights,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            apply_transforms: bool,
    ) -> None:
        self.features = features
        self.responses = responses
        self.weights = weights
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.batch_size = batch_size
        self.apply_transforms = apply_transforms

        # TODO: Phil:  what does this do exactly?
        self.cum_samples_per_fold = np.zeros(len(features)+1).astype(int)
        for fold in range(1, len(features)+1):
            self.cum_samples_per_fold[fold] = features[fold-1].shape[0] + self.cum_samples_per_fold[fold-1]

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        return int(np.ceil(self.cum_samples_per_fold[-1] / self.batch_size))

    def _get_features_responses_weights(self, index: int) -> Tuple[np.array, np.array, np.array]:
        # TODO:  Phil:  could you comment so it's easier to see what's going on here?
        current_fold = 0
        while current_fold < len(self.cum_samples_per_fold) - 1:
            if ((index * self.batch_size >= self.cum_samples_per_fold[current_fold] and
                 index * self.batch_size < self.cum_samples_per_fold[current_fold+1])):
                break
            current_fold += 1
            q = 1

        sample_index = int(index * self.batch_size - self.cum_samples_per_fold[current_fold])

        batch_features = (self.features[current_fold])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_responses = (self.responses[current_fold])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_weights = (self.weights[current_fold])[sample_index:sample_index+self.batch_size, ...].copy()

        while (batch_features.shape[0] < self.batch_size):
            sample_index = 0
            current_fold += 1

            if (current_fold == len(self.features)):
                break

            stop_ind = self.batch_size - batch_features.shape[0]
            batch_features = np.append(batch_features, (self.features[current_fold])[
                                       sample_index:stop_ind, ...], axis=0)
            batch_responses = np.append(batch_responses, (self.responses[current_fold])[
                                        sample_index:stop_ind, ...], axis=0)
            batch_weights = np.append(batch_weights, (self.weights[current_fold])[sample_index:stop_ind, ...], axis=0)
        return batch_features, batch_responses, batch_weights
