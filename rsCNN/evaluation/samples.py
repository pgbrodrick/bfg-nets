from typing import Tuple

import keras
import numpy as np

from rsCNN.data_management import scalers, sequences


class Samples(object):
    num_samples = None
    num_features = None
    num_responses = None
    data_sequence = None
    raw_features = None
    raw_features_range = None
    trans_features = None
    trans_features_range = None
    raw_responses = None
    raw_responses_range = None
    trans_responses = None
    trans_responses_range = None
    raw_predictions = None
    raw_predictions_range = None
    trans_predictions = None
    trans_predictions_range = None
    weights = None
    weights_range = None

    def __init__(self, data_sequence: sequences.BaseSequence, model: keras.Model) -> None:
        self.data_sequence = data_sequence
        sampled_features, sampled_responses = self._get_sampled_features_responses_and_set_metadata_and_weights()
        predictions = model.predict(sampled_features)
        self._set_raw_and_transformed_features_and_responses(sampled_features, sampled_responses, predictions)

    def _get_sampled_features_responses_and_set_metadata_and_weights(self) -> Tuple[np.array, np.array]:
        # TODO:  handle getting representative samples
        # TODO:  handle multiple inputs
        features, responses = self.data_sequence.__getitem__(0)
        # We expect weights to be the last element in the responses array
        self.weights = responses[0][..., -1]
        # TODO:  confirm this was not necessary:
        # weights = weights.reshape((weights.shape[0], weights.shape[1], weights.shape[2], 1))
        # Unpack features and responses, convert nodata values to nan
        features = features[0]
        responses = responses[0][..., :-1]
        features[features == self.data_sequence.feature_scaler.nodata_value] = np.nan
        responses[responses == self.data_sequence.response_scaler.nodata_value] = np.nan
        # Set sample metadata
        self.num_samples = features.shape[0]
        self.num_features = features.shape[-1]
        self.num_responses = responses.shape[-1]
        return features, responses

    def _set_raw_and_transformed_features_and_responses(
            self,
            sampled_features: np.array,
            sampled_responses: np.array,
            predictions: np.array
    ) -> None:
        if type(self.data_sequence.feature_scaler) is scalers.NullScaler:
            self.raw_features = sampled_features
            self.trans_features = None
        else:
            self.raw_features = self.data_sequence.feature_scaler.inverse_transform(sampled_features)
            self.trans_features = sampled_features
        if type(self.data_sequence.response_scaler) is scalers.NullScaler:
            self.raw_responses = sampled_responses
            self.trans_responses = None
            self.raw_predictions = predictions
            self.trans_predictions = None
        else:
            self.raw_responses = self.data_sequence.response_scaler.inverse_transform(sampled_responses)
            self.trans_responses = sampled_responses
            self.raw_predictions = self.data_sequence.response_scaler.inverse_transform(predictions)
            self.trans_predictions = predictions

    def _set_raw_and_transformed_ranges(self):
        # Note:  ranges are a (num_values, 2) array, e.g., raw_features_range with 3 features is a (3, 2) array with
        # raw_features_range[2, 0] being the minimum of the 3rd feature and raw_features_range[2, 1] being the maximum
        self.raw_features_range = self._get_range(self.raw_features)
        self.trans_features_range = self._get_range(self.trans_features)
        self.raw_responses_range = self._get_range(self.raw_responses)
        self.trans_responses_range = self._get_range(self.trans_responses)
        self.raw_predictions_range = self._get_range(self.raw_predictions)
        self.trans_predictions_range = self._get_range(self.trans_predictions)
        self.weights_range = self._get_range(self.weights)

    def _get_range(self, data: np.array) -> np.array:
        mins = np.nanpercentile(data.reshape((-1, data.shape[-1])), 0, axis=0)
        maxs = np.nanpercentile(data.reshape((-1, data.shape[-1])), 100, axis=0)
        return np.dstack([mins, maxs])[0]
