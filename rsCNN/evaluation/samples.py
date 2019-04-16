from typing import Tuple

import keras
import numpy as np

from rsCNN.data_management import scalers, sequences


class Samples(object):
    data_sequence = None
    raw_features = None
    trans_features = None
    raw_responses = None
    trans_responses = None
    raw_predictions = None
    trans_predictions = None
    weights = None

    def __init__(self, data_sequence: sequences.BaseSequence, model: keras.Model) -> None:
        self.data_sequence = data_sequence
        sampled_features, sampled_responses = self._get_sampled_features_responses_and_set_weights()
        predictions = model.predict(sampled_features)
        self._set_raw_and_transformed_features_and_responses(sampled_features, sampled_responses, predictions)

    def _get_sampled_features_responses_and_set_weights(self) -> Tuple[np.array, np.array]:
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
