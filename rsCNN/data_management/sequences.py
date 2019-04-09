import random
from typing import List, Tuple

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.utils import logger
from rsCNN.data_management import sequences

_logger = logger.get_child_logger(__name__)


# TODO: this is almost certainly the wrong place for this
def build_memmaped_sequence(data_config, fold_indices, batch_size = 100, rebuild = False):

        """
            This function does the following, considering the rebuild parameter at each step:
                2) load/initialize/fit scalers
                3) initiate train/validation/test sequences as components of Experiment
        """

        assert data_config.features is not None, 'data config must have loaded feature numpy files'
        assert data_config.responses is not None, 'data config must have loaded responses numpy files'
        assert data_config.weights is not None, 'data config must have loaded weight numpy files'

        assert data_config.feature_scaler is not None, 'Feature scaler must be defined'
        assert data_config.response_scaler is not None, 'Response scaler must be defined'


        apply_random = data_config['training']['apply_random_transformations']
        mean_centering = data_config.feature_mean_centering
        data_sequence = sequences.MemmappedSequence([data_config.features[_f] for _f in fold_indices],
                                                    [data_config.responses[_r] for _r in fold_indices],
                                                    [data_config.weights[_w] for _w in fold_indices],
                                                    data_config.feature_scaler,
                                                    data_config.response_scaler,
                                                    batch_size,
                                                    apply_random_transforms=apply_random,
                                                    feature_mean_centering=mean_centering)
        return data_sequence

class BaseSequence(keras.utils.Sequence):
    feature_scaler = None
    response_scaler = None
    apply_random_transforms = None

    def __init__(
            self,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            apply_random_transforms: bool = False,
            nan_conversion_value: float = None
    ) -> None:
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.batch_size = batch_size
        self.apply_random_transforms = apply_random_transforms
        self.nan_conversion_value = nan_conversion_value

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

    def _mean_center(self, data: List[np.array]) -> List[np.array]:
        return data - np.mean(data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

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
            feature_mean_centering: False,
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

        self.feature_mean_centering = feature_mean_centering

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
        if (self.feature_mean_centering is True):
            batch_features = self._mean_center(batch_features)
        return [batch_features], [batch_responses], [batch_weights]
