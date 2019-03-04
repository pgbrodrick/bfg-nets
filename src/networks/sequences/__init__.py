import random

import keras
import numpy as np

from rsCNN.utils import logging


_logger = logging.get_child_logger(__name__)


class BaseSequence(keras.utils.Sequence):
    batches_per_epoch = None

    def __init__(self, batches_per_epoch=None):
        self.batches_per_epoch = batches_per_epoch

    def __len__(self):
        # Method is required for Keras functionality
        return self.batches_per_epoch  # a.k.a. steps_per_epoch in fit_generator

    def __getitem__(self, item):
        # Method is required for Keras functionality
        raise NotImplementedError


class BaseCNNSequence(BaseSequence):
    architecture = None
    data_sampler = None
    data_transformer = None
    geospatial_index = None
    batch_size = None
    image_dim = None
    num_timesteps = None
    testing_indices = None
    type = None

    def __init__(self, data_sampler, data_transformer, batch_size=None, batches_per_epoch=None, image_dim=None,
                 num_timesteps=None):
        assert self.architecture in (_ARCH_PREDICTION, _ARCH_RECONSTRUCTION, _ARCH_PREDICTION_RECONSTRUCTION), \
            'Architecture must be set correctly'
        self.data_sampler = data_sampler
        self.data_transformer = data_transformer
        self.batch_size = batch_size
        self.image_dim = image_dim
        self.num_timesteps = num_timesteps
        super().__init__(batches_per_epoch)

    def __getitem__(self, idx):
        _logger.debug('Get {} batch with {} items via sequence'.format(self.type.lower(), self.batch_size))
        if idx == 0:
            self.data_sampler.set_new_epoch()  # If new epoch, used to avoid using bootstrapped samples multiple times
        batch_inputs = None
        batch_targets = None
        for idx_pair in range(self.batch_size):
            # TODO:  the sampler needs to return arrays of inputs and targets, where inputs = [x0, x1, x2 .. xN], etc,
            # TODO:  and where the merge and transform steps assume that x and y come as arrays. More importantly, I
            # TODO:  think we might want to change how samples are handled, since this pipeline is a bit gross and
            # TODO:  hacky.
            x, y = self.data_sampler.get_sample(self.image_dim, self.num_timesteps)['sample']
            x = self._merge_x_timesteps_and_channels(x)
            x, y = self._transform_x_y(x, y)
            # Set input and target list lengths after seeing the dimension of inputs and targets
            if batch_inputs is None or batch_targets is None:
                batch_inputs = [list() for _ in x]
                batch_targets = [list() for _ in y]
            # Append inputs and targets to the appropriate lists
            for input_, inputs in zip(x, batch_inputs):
                inputs.append([input_])
            for target, targets in zip(y, batch_targets):
                targets.append([target])
        _logger.debug('Retrieved full batch'.format(self.batch_size))
        # Need list of np arrays for *_generator model methods
        batch_inputs = [np.array(inputs) for inputs in batch_inputs]
        batch_targets = [np.array(targets) for targets in batch_targets]
        return batch_inputs, batch_targets

    def _merge_x_timesteps_and_channels(self, x):
        # Start with (num_timesteps, image_dim, image_dim, num_channels)
        # Moveaxis to (image_dim, image_dim, num_timesteps, num_channels)
        # Reshape (image_dim, image_dim, num_timesteps * num_channels)
        # Note the last axis will be (b0, g0, r0, b1, g1, r1 ... bn, gn, rn)
        # Note that num_timesteps may be == 1 and this still works
        return np.moveaxis(x, 0, 2).reshape(self.image_dim, self.image_dim, -1)

    def _transform_x_y(self, x, y):
        # Segment images for better fit
        # TODO:  figure out whether we want this included and how to better handle?
        if self.is_segmented is True:
            x, y = self._segment_x_y(x, y)
        # Required transforms for data quality
        x = self.data_transformer.transform(x)
        y = self.data_transformer.transform(y)
        # TODO:  figure out how to better handle training (transform) vs testing (no transform) and how to better name
        # TODO:  variables/modules to distinguish between pre-processing transforms and on-the-fly training variability
        # TODO:  transforms
        if self.type is not _TYPE_TRAINING:
            return x, y
        # Optional transforms for training variety
        # Flip top to bottom
        if random.random() > 0.5:
            x = np.flip(x, axis=0)
            y = np.flip(y, axis=0)
        # Flip side to side
        if random.random() > 0.5:
            x = np.flip(x, axis=1)
            y = np.flip(y, axis=1)
        # Rotate 0, 1, 2, or 3 times
        num_rotations = np.floor(4 * random.random())
        x = np.rot90(x, k=num_rotations, axes=(0, 1))
        y = np.rot90(y, k=num_rotations, axes=(0, 1))
        return x, y

    def _segment_x_y(self, x, y):
        num_x_images = x.shape[2] / self.num_timesteps
        for idx_image in range(num_x_images):
            idx_start = 2 * idx_image
            idx_finish = idx_start + 2
            x[:, :, idx_start:idx_finish] = self._segment_bands(x[:, :, idx_start:idx_finish])
        y = self._segment_bands(y)
        return x, y

    def _segment_bands(self, bands):
        # Not really RGB, just need the right dimensions
        # TODO:  I can provide this code if we include segmentations ... I don't think we'll want this
        rgb_image = np.dstack([bands, np.zeros((bands.shape[0], bands.shape[1], 1))])
        segments = segmentation.generate_segmentation_felzenszwalb(rgb_image)
        idx_missing = ~np.all(np.isfinite(rgb_image), axis=2)
        segments = segmentation.remove_segments_for_missing_data(segments, idx_missing)
        return segmentation.calculate_segmentation_image(rgb_image, segments)[:, :, :2]
