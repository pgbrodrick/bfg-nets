import keras
import numpy as np

from rsCNN.configuration import configs
from rsCNN.data_management import scalers, sequences


class Samples(object):
    data_sequence = None
    data_sequence_label = None
    model = None
    config = None
    num_samples = None
    num_features = None
    num_responses = None
    has_features_transform = None
    has_responses_transform = None
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

    def __init__(
            self,
            data_sequence: sequences.BaseSequence,
            model: keras.Model,
            config: configs.Config,
            data_sequence_label: str = None
    ) -> None:
        self.data_sequence = data_sequence
        self.model = model
        self.config = config
        self.data_sequence_label = data_sequence_label
        self._get_sampled_features_responses_and_set_metadata_and_weights()
        self.trans_predictions = model.predict(self.trans_features)
        self.raw_predictions = self.data_sequence.response_scaler.inverse_transform(self.trans_predictions)
        self._set_has_transforms()
        self._set_raw_and_transformed_ranges()

    def _get_sampled_features_responses_and_set_metadata_and_weights(self) -> None:
        # TODO:  handle getting representative samples, e.g., get images that show specific classes so all are covered,
        #  probably want to do something like, given x classes, for each class find y images with class in the loss
        #  window, for x * y images total
        # TODO:  handle multiple inputs when necessary
        (raw_features, raw_responses), (trans_features, trans_responses) = \
            self.data_sequence.get_raw_and_transformed_sample(0)
        # We expect weights to be the last element in the responses array
        self.weights = trans_responses[0][..., -1]
        # Unpack features and responses, inverse transform to get raw values
        self.raw_features = raw_features[0]
        self.trans_features = trans_features[0]
        self.raw_responses = raw_responses[0][..., :-1]
        self.trans_responses = trans_responses[0][..., :-1]
        # Set sample metadata
        self.num_samples = self.trans_features.shape[0]
        self.num_features = self.trans_features.shape[-1]
        self.num_responses = self.trans_responses.shape[-1]

    def _set_has_transforms(self) -> None:
        if type(self.data_sequence.feature_scaler) is scalers.NullScaler:
            self.has_features_transform = False
        else:
            self.has_features_transform = True
        if type(self.data_sequence.response_scaler) is scalers.NullScaler:
            self.has_responses_transform = False
        else:
            self.has_responses_transform = True

    def _set_raw_and_transformed_ranges(self):
        # Note:  ranges are a (num_values, 2) array, e.g., raw_features_range with 3 features is a (3, 2) array with
        # raw_features_range[2, 0] being the minimum of the 3rd feature and raw_features_range[2, 1] being the maximum
        self.raw_features_range = self._get_range(self.raw_features)
        tmp_features = self.trans_features.copy()
        tmp_features[tmp_features == self.data_sequence.nan_replacement_value] = np.nan
        self.trans_features_range = self._get_range(tmp_features)

        self.raw_responses_range = self._get_range(self.raw_responses)
        tmp_responses = self.trans_responses.copy()
        tmp_responses[tmp_responses == self.data_sequence.nan_replacement_value] = np.nan
        self.trans_responses_range = self._get_range(tmp_responses)

        self.raw_predictions_range = self._get_range(self.raw_predictions)
        tmp_predictions = self.trans_predictions.copy()
        tmp_predictions[tmp_predictions == self.data_sequence.nan_replacement_value] = np.nan
        self.trans_predictions_range = self._get_range(tmp_predictions)

        tmp_weights = self.weights.copy()
        tmp_weights[tmp_weights == 0] = np.nan
        self.weights_range = self._get_range(tmp_weights)

    def _get_range(self, data: np.array) -> np.array:
        if data is None:
            return None
        last_dim = data.shape[-1]
        if len(data.shape) == 3:
            last_dim = 1
        mins = np.nanpercentile(data.reshape((-1, last_dim)), 0, axis=0)
        maxs = np.nanpercentile(data.reshape((-1, last_dim)), 100, axis=0)
        return np.dstack([mins, maxs])[0]
