import logging
import random
from typing import List, Tuple

import keras
import numpy as np

from rsCNN.data_management.scalers import BaseGlobalScaler
from rsCNN.data_management.training_data import Dataset


_logger = logging.getLogger(__name__)


# TODO: this is almost certainly the wrong place for this
def build_memmapped_sequence(data_container: Dataset, fold_indices, batch_size=100, rebuild=False):
    """
        This function does the following, considering the rebuild parameter at each step:
            2) load/initialize/fit scalers
            3) initiate train/validation/test sequences as components of Experiment
    """

    assert data_container.features is not None, 'data_container must have loaded feature numpy files'
    assert data_container.responses is not None, 'data_container must have loaded responses numpy files'
    assert data_container.weights is not None, 'data_container must have loaded weight numpy files'

    assert data_container.feature_scaler is not None, 'Feature scaler must be defined'
    assert data_container.response_scaler is not None, 'Response scaler must be defined'

    data_sequence = MemmappedSequence(
        [data_container.features[_f] for _f in fold_indices],
        [data_container.responses[_r] for _r in fold_indices],
        [data_container.weights[_w] for _w in fold_indices],
        data_container.feature_scaler,
        data_container.response_scaler,
        batch_size,
        apply_random_transforms=data_container.config.data_samples.apply_random_transformations,
        feature_mean_centering=data_container.config.data_build.feature_mean_centering,
        nan_replacement_value=data_container.config.data_samples.feature_nodata_encoding
    )
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
            nan_replacement_value: float = None
    ) -> None:
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.batch_size = batch_size
        self.apply_random_transforms = apply_random_transforms
        self.nan_replacement_value = nan_replacement_value

    def __len__(self) -> int:
        raise NotImplementedError('Method is required for Keras functionality. Should return steps_per_epoch.')

    def __getitem__(self, index: int) -> Tuple[List[np.array], List[np.array]]:
        # Method is required for Keras functionality
        _logger.debug('Get batch {} with {} items via sequence'.format(index, self.batch_size))
        features, responses, weights = self._get_features_responses_weights(index)
        return self._get_transformed_sample(features, responses, weights)

    def get_raw_and_transformed_sample(self, index: int) -> \
            Tuple[Tuple[List[np.array], List[np.array]], Tuple[List[np.array], List[np.array]]]:
        _logger.debug('Get batch {} with {} items via sequence'.format(index, self.batch_size))
        _logger.debug('Get features, responses, and weights')
        raw_features, raw_responses, raw_weights = self._get_features_responses_weights(index)
        trans_features, trans_responses = self._get_transformed_sample(
            raw_features.copy(), raw_responses.copy(), raw_weights.copy()
        )
        raw_responses = [np.append(response, weight, axis=-1) for response, weight in zip(raw_responses, raw_weights)]
        return ((raw_features, raw_responses), (trans_features, trans_responses))

    def _get_transformed_sample(self, raw_features, raw_responses, raw_weights) -> \
            Tuple[List[np.array], List[np.array]]:
        _logger.debug('Optionally modify features, responses, and weights prior to scaling')
        # Reusing names to avoid creating new, large objects
        raw_features, raw_responses, raw_weights = \
            self._modify_features_responses_weights_before_scaling(raw_features, raw_responses, raw_weights)
        _logger.debug('Scale features')
        raw_features = self._scale_features(raw_features)
        _logger.debug('Scale responses')
        raw_responses = self._scale_responses(raw_responses)
        if self.nan_replacement_value is not None:
            _logger.debug('Convert nan features to {}'.format(self.nan_replacement_value))
            raw_features = self._replace_nan_data_values(raw_features, self.nan_replacement_value)
            _logger.debug('Convert nan responses to {}'.format(self.nan_replacement_value))
            raw_responses = self._replace_nan_data_values(raw_responses, self.nan_replacement_value)
        else:
            assert np.all(np.isfinite(raw_features)), \
                'Some feature values are nan but nan_replacement_value not provided in data config. Please provide ' + \
                'a nan_replacement_value to transform features correctly.'
        _logger.debug('Append weights to responses for loss function calculations')
        raw_responses = [np.append(response, weight, axis=-1) for response, weight in zip(raw_responses, raw_weights)]
        if self.apply_random_transforms is True:
            _logger.debug('Apply random transformations to features and responses')
            self._apply_random_transformations(raw_features, raw_responses)
        else:
            _logger.debug('Random transformations not applied to features and responses')
        return raw_features, raw_responses

    def _get_features_responses_weights(self, index: int) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        raise NotImplementedError(
            'Custom Sequences must implement _get_features_responses_weights for training and reporting to work. ' +
            'See method header for expected arguments and returned objects.'
        )

    def _replace_nan_data_values(self, data: List[np.array], replacement_value):
        for idx_array in range(len(data)):
            data[idx_array][np.isnan(data[idx_array])] = replacement_value
        return data

    def _modify_features_responses_weights_before_scaling(
            self,
            features: List[np.array],
            responses: List[np.array],
            weights: List[np.array]
    ) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        _logger.debug('No preliminary modifications applied to features, responses, or weights')
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
            feature_mean_centering: False,
            nan_replacement_value: None,
    ) -> None:
        self.features = features  # a list of numpy arrays, each of which is (n,y,x,f)
        self.responses = responses  # a list of numpy arrays, each of which is (n,y,x,r)
        self.weights = weights  # a list of numpy arrays, each of which is (n,y,x,1)
        super().__init__(feature_scaler=feature_scaler, response_scaler=response_scaler, batch_size=batch_size,
                         apply_random_transforms=apply_random_transforms, nan_replacement_value=nan_replacement_value)

        # Determine the cumulative number of total samples across arrays - we're going to use
        # it to roll between files when extracting samples
        self.cum_samples_per_array = np.zeros(len(features)+1).astype(int)
        for _array in range(1, len(features)+1):
            self.cum_samples_per_array[_array] = features[_array-1].shape[0] + self.cum_samples_per_array[_array-1]

        self.feature_mean_centering = feature_mean_centering

    def __len__(self):
        # Method is required for Keras functionality, a.k.a. steps_per_epoch in fit_generator
        return int(np.ceil(self.cum_samples_per_array[-1] / self.batch_size))

    def _mean_center(self, data: np.array) -> np.array:
        return data - np.mean(data, axis=(1, 2))[:, np.newaxis, np.newaxis, :]

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
