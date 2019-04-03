import random
from typing import List, Tuple

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.utils import logger


_logger = logger.get_child_logger(__name__)


class BaseSequence(keras.utils.Sequence):
    feature_scaler = None
    response_scaler = None
    apply_random_transforms = None

    def __init__(
            self,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            apply_random_transforms: bool = False
    ) -> None:
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.batch_size = batch_size
        self.apply_random_transforms = apply_random_transforms

    def __len__(self) -> int:
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[List[np.array], List[np.array]]:
        # Method is required for Keras functionality
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
        responses_with_weights = [np.append(response, weight, axis=-1) for response, weight in zip(responses, weights)]
        if self.apply_random_transforms is True:
            _logger.debug('Apply random transformations to features and responses')
            self._apply_random_transformations(features, responses)
        return features, responses_with_weights

    def _get_features_responses_weights(self, index: int) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        raise NotImplementedError

    def _modify_features_responses_weights_before_scaling(
            self,
            features: List[np.array],
            responses: List[np.array],
            weights: List[np.array]
    ) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        return features, responses, weights

    def _scale_features(self, features: List[np.array]) -> List[np.array]:
        return [self.feature_scaler.transform(feature) for feature in features]

    def _scale_responses(self, responses: List[np.array]) -> List[np.array]:
        return [self.response_scaler.transform(response) for response in responses]

    def _apply_random_transformations(
            self,
            features: List[np.array],
            responses: [np.array]
    ) -> Tuple[np.array, np.array]:
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

    def __init__(
            self,
            features,
            responses,
            weights,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            apply_random_transforms: bool,
    ) -> None:
        self.features = features  # a list of numpy arrays, each of which is (n,y,x,f)
        self.responses = responses  # a list of numpy arrays, each of which is (n,y,x,r)
        self.weights = weights  # a list of numpy arrays, each of which is (n,y,x,1)
        super().__init__(feature_scaler=feature_scaler, response_scaler=response_scaler, batch_size=batch_size,
                         apply_random_transforms=apply_random_transforms)

        # Determine the cumulative number of total samples across arrays - we're going to use
        # it to roll between files when exctracting samples
        self.cum_samples_per_array = np.zeros(len(features)+1).astype(int)
        for _array in range(1, len(features)+1):
            self.cum_samples_per_array[_array] = features[_array-1].shape[0] + self.cum_samples_per_array[_array-1]

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        return int(np.ceil(self.cum_samples_per_array[-1] / self.batch_size))

    def _get_features_responses_weights(self, index: int) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        # start by finding which array we're starting in, based on the input index, batch size,
        # and the number of samples per array
        current_array = 0
        while current_array < len(self.cum_samples_per_array) - 1:
            if ((index * self.batch_size >= self.cum_samples_per_array[current_array] and
                 index * self.batch_size < self.cum_samples_per_array[current_array+1])):
                break
            current_array += 1
            q = 1

        # grab the the appropriate number of samples from the current array
        sample_index = int(index * self.batch_size - self.cum_samples_per_array[current_array])

        batch_features = (self.features[current_array])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_responses = (self.responses[current_array])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_weights = (self.weights[current_array])[sample_index:sample_index+self.batch_size, ...].copy()

        # if the current array didn't have enough samples in it, roll forward to the next one (and keep
        # doing so until we have enough samples)
        # TODO: probably need a safety here so we terminate at the point where we hit the first sample grabbed....only applies for very small datasets or very large batch_sizes, but we shoudl safeguard anyway.
        while (batch_features.shape[0] < self.batch_size):
            sample_index = 0
            current_array += 1

            if (current_array == len(self.features)):
                break

            stop_ind = self.batch_size - batch_features.shape[0]
            batch_features = np.append(batch_features, (self.features[current_array])[
                                       sample_index:stop_ind, ...], axis=0)
            batch_responses = np.append(batch_responses, (self.responses[current_array])[
                                        sample_index:stop_ind, ...], axis=0)
            batch_weights = np.append(batch_weights, (self.weights[current_array])[sample_index:stop_ind, ...], axis=0)
        return [batch_features], [batch_responses], [batch_weights]
