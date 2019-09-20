import logging
from typing import List, Tuple, Union

import albumentations
import keras
import numpy as np

from bfgn.data_management.scalers import BaseGlobalScaler


_logger = logging.getLogger(__name__)


ADDITIONAL_TARGETS_KEY = 'image_{}'


class BaseSequence(keras.utils.Sequence):
    feature_scaler = None
    response_scaler = None
    custom_augmentations = None

    def __init__(
            self,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            custom_augmentations: albumentations.Compose = None,
            nan_replacement_value: float = None
    ) -> None:
        self.feature_scaler = feature_scaler
        self.response_scaler = response_scaler
        self.batch_size = batch_size
        self.custom_augmentations = custom_augmentations
        self.nan_replacement_value = nan_replacement_value

    def __len__(self) -> int:
        raise NotImplementedError('Method is required for Keras functionality. Should return steps_per_epoch.')

    def __getitem__(self, index: int, return_raw_sample: bool = False) -> \
            Union[
                Tuple[List[np.array], List[np.array]],
                Tuple[Tuple[List[np.array], List[np.array]], Tuple[List[np.array], List[np.array]]],
            ]:
        # Method is required for Keras functionality, reuse names to avoid creating many new, potentially large objects
        _logger.debug('Get batch {} with {} items via sequence'.format(index, self.batch_size))
        raw_features, raw_responses, raw_weights = self._get_features_responses_weights(index)

        trans_features = [raw_feature.copy() for raw_feature in raw_features]
        trans_responses = [raw_response.copy() for raw_response in raw_responses]
        trans_weights = [raw_weight.copy() for raw_weight in raw_weights]

        if self.custom_augmentations is not None:
            trans_features, trans_responses, trans_weights = \
                self._apply_augmentations(trans_features, trans_responses, trans_weights)

        trans_features = self._scale_features(trans_features)
        trans_responses = self._scale_responses(trans_responses)

        if self.nan_replacement_value is not None:
            raw_features = self._replace_nan_data_values(raw_features, self.nan_replacement_value)
            raw_responses = self._replace_nan_data_values(raw_responses, self.nan_replacement_value)
        else:
            assert np.all(np.isfinite(raw_features)), \
                'Some feature values are nan but nan_replacement_value not provided in data config. Please provide ' + \
                'a nan_replacement_value to transform features correctly.'

        # Append weights to responses for loss function calculations
        raw_responses = [np.append(response, weight, axis=-1) for response, weight in zip(raw_responses, raw_weights)]
        trans_responses = [np.append(resp, weight, axis=-1) for resp, weight in zip(trans_responses, trans_weights)]

        if return_raw_sample is True:
            # This is for BGFN reporting and other functionality
            return_value = ((raw_features, raw_responses), (trans_features, trans_responses))
        else:
            # This is for Keras sequence generator behavior
            return_value = (trans_features, trans_responses)
        return return_value

    def get_raw_and_transformed_sample(self, index: int) -> \
            Tuple[Tuple[List[np.array], List[np.array]], Tuple[List[np.array], List[np.array]]]:
        return self.__getitem__(index, return_raw_sample=True)

    def _get_features_responses_weights(self, index: int) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        raise NotImplementedError(
            'Custom Sequences must implement _get_features_responses_weights for training and reporting to work. ' +
            'See method header for expected arguments and returned objects.'
        )

    def _replace_nan_data_values(self, data: List[np.array], replacement_value):
        for idx_array in range(len(data)):
            data[idx_array][np.isnan(data[idx_array])] = replacement_value
        return data

    def _apply_augmentations(
        self,
        features: List[np.array],
        responses: List[np.array],
        weights: List[np.array]
    ) -> Tuple[List[np.array], List[np.array], List[np.array]]:
        assert len(responses) == 1, \
            'Custom augmentations have not been tested on multiple responses. Please feel free to handle this ' + \
            'case, test your code, and submit a pull request.'
        # Loop through samples, augmenting each
        num_samples = features[0].shape[0]
        for idx_sample in range(num_samples):
            # Get sample data
            sample_features = [feature[idx_sample] for feature in features]
            sample_responses = responses[0][idx_sample]  # Assume single response
            sample_weights = weights[0][idx_sample]
            mask_loss_window = (sample_weights > 0)[..., 0]
            # Format for albumentations.Compose
            data_to_augment = {'image': sample_features.pop(0), 'mask': np.dstack([sample_responses, sample_weights])}
            target_keys = ['image']
            for idx, feature in enumerate(sample_features):
                key_feature = ADDITIONAL_TARGETS_KEY.format(idx+1)
                data_to_augment[key_feature] = feature
                target_keys.append(key_feature)
            # Augment data and parse results
            augmented = self.custom_augmentations(**data_to_augment)
            sample_features = list()  # For creating a weights mask
            for idx_feature, key_feature in enumerate(target_keys):
                features[idx_feature][idx_sample] = augmented[key_feature]
                sample_features.append(augmented[key_feature])
            responses[0][idx_sample] = augmented['mask'][..., :-1]
            mask_features = np.isfinite(np.dstack(sample_features)).all(axis=-1)
            mask = np.logical_and(mask_features, mask_loss_window)
            weights[0][idx_sample] = np.expand_dims(mask * augmented['mask'][..., -1], axis=-1)
        return features, responses, weights

    def _scale_features(self, features: List[np.array]) -> List[np.array]:
        return [self.feature_scaler.transform(feature) for feature in features]

    def _scale_responses(self, responses: List[np.array]) -> List[np.array]:
        return [self.response_scaler.transform(response) for response in responses]


class MemmappedSequence(BaseSequence):

    def __init__(
            self,
            features,
            responses,
            weights,
            feature_scaler: BaseGlobalScaler,
            response_scaler: BaseGlobalScaler,
            batch_size: int,
            feature_mean_centering: False,
            nan_replacement_value: None,
            custom_augmentations: albumentations.Compose = None,
    ) -> None:
        self.features = features  # a list of numpy arrays, each of which is (n,y,x,f)
        self.responses = responses  # a list of numpy arrays, each of which is (n,y,x,r)
        self.weights = weights  # a list of numpy arrays, each of which is (n,y,x,1)
        super().__init__(
            feature_scaler=feature_scaler, response_scaler=response_scaler, batch_size=batch_size,
            custom_augmentations=custom_augmentations, nan_replacement_value=nan_replacement_value
        )

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

        # grab the the appropriate number of samples from the current array
        sample_index = int(index * self.batch_size - self.cum_samples_per_array[current_array])

        batch_features = (self.features[current_array])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_responses = (self.responses[current_array])[sample_index:sample_index+self.batch_size, ...].copy()
        batch_weights = (self.weights[current_array])[sample_index:sample_index+self.batch_size, ...].copy()

        # if the current array didn't have enough samples in it, roll forward to the next one (and keep
        # doing so until we have enough samples)
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


def sample_custom_augmentations_constructor(num_features: int, window_radius: int) -> albumentations.Compose:
    """
    This function returns a custom augmentations object for use with sequences via the load_sequences function in
    data_core.py. Please note that these augmentations have only been tested with RGB data between 0 and 1 and that
    order of operations is critical. e.g., blurs don't like missing data so shouldn't be applied before dropout, noise
    probably shouldn't be applied before color changes or blurs... of course, this is all dependent on your specific
    problem.

    Args:
        num_features:  number of features used in the model
        window_size:  window_size from the data configs

    Returns:
        custom augmentations function for use with sequences
    """
    max_kernel = int(round(0.1 * window_radius))
    max_hole_size = int(round(0.1 * window_radius))
    additional_targets = [ADDITIONAL_TARGETS_KEY.format(idx) for idx in range(1, num_features)]

    return albumentations.Compose([
        # The augmentations assume an image is RGB between 0 and 1
        albumentations.ToFloat(max_value=255, always_apply=True, p=1.0),

        # These augmentations should be order independent, toss 'em up front
        albumentations.Flip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.Rotate(limit=90, p=0.5),

        # Fogging as it's quite similar to top-down cloud effects, seems reasonable to apply up front
        albumentations.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.8, alpha_coef=0.08, p=0.5),

        # Color modifications
        albumentations.OneOf([
            albumentations.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.6, brightness_by_max=True, p=1.0),
            albumentations.RGBShift(r_shift_limit=0.2, g_shift_limit=0.2, b_shift_limit=0.2, p=1.0),
        ], p=0.25),

        # Distortions
        albumentations.OneOf([
            albumentations.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=0.4, p=1.0),
            albumentations.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=1.0),
        ], p=0.25),
        albumentations.GaussianBlur(blur_limit=max_kernel, p=0.25),

        # Noise
        albumentations.OneOf([
            albumentations.CoarseDropout(
                max_holes=8, max_height=max_hole_size, max_width=max_hole_size, fill_value=np.nan, p=1.0),
            albumentations.GaussNoise(var_limit=0.05, mean=0, p=1.0),
        ], p=0.25),

        # Scaling, adding last so that other augmentations are applied at a consistent resolution
        albumentations.RandomScale(scale_limit=0.05, p=0.25),

        # Augmentations may not return images of the same size, images can be both smaller and larger than expected, so
        # these two augmentations are added to keep things consistent
        albumentations.PadIfNeeded(2*window_radius, 2*window_radius, always_apply=True, p=1.0),
        albumentations.CenterCrop(2*window_radius, 2*window_radius, always_apply=True, p=1.0),

        # Return the data to its original scale
        albumentations.FromFloat(max_value=255, always_apply=True, p=1.0),
    ], p=1.0, additional_targets={target: 'image' for target in additional_targets})
