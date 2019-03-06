import datetime
import os
import pickle

import keras
import keras.backend as K
import numpy as np

import src.networks.networks



class network_config:
  """ A wrapper class designed to hold all relevant configuration information for the
      training of a new network.  
  """ 
  def __init__(self,network_type,inshape,n_classes):
    """ 
      Arguments:
      network_type - str
        Style of the network to use.  Options are:
          flex_unet
          flat_regress_net
      inshape - tuple/list
        Designates the input shape of an image to be passed to
        the network.
      n_classes - int
        The number of classes the network is meant to classify/regress.
    """ 
    self.network_type = network_type
    self.inshape = inshape
    self.n_classes = n_classes

    #TODO: Fabina, if there are other things you want to be read in (defaults like directories
    #or whatnot, add them here as defaults.  The read_from* functions can pull in overwrite values
    #as desired

    self.option = {}
   
  def read_from_file(filename):
    """ Read in optional arguments from file
    Arguments:
    filename - str
      Name of file to read.
    """

    ################ TODO: Read in from file (yaml?)
    a = None

  def read_from_dict(in_dict):
    """ Read in optional arguments from dictionary
    Arguments:
    in_dict - dictionary
      Dictionary of values to read in.
    """
    for key in in_dict:
      self.option[key] = in_dict[key]


######## TODO - Fabina, can you populate this with the useful info you want to retain from training?
class training_history:
  """ A wrapper class designed to hold all relevant configuration information obtained
      during training/testing the model.  
  """ 


class CNN():

  def __init__(self):
    self.model = None
    self.config = None
    self.training = None


  def create_config(network_name,input_shape,n_classes,network_file=None,network_dictionary=None):
    self.config = network_config(network_name,input_shape,n_classes)

    if (network_file is not None):
      self.config.read_from_file(network_file)

    if (network_dictionary is not None):
      self.config.read_from_dict(network_dictionary)


  def create_network():
    """ Initializes the appropriate network

    Arguments:
    net_name - str
      Name of the network to fetch.  Options are:
        flex_unet - a flexible, U-net style network.
    inshape - tuple/list
      Designates the input shape of an image to be passed to
      the network.
    n_classes - int
      The number of classes the network is meant to classify.
    """

    if (self.network_name == 'flex_unet'):
      # Update potentially non-standard network parameters
      self.config.set_default('conv_depth',16)
      self.config.set_default('batch_norm',False)

      # Return a call to the argument-specific version of flex_unet
      self.model =  flex_unet(self.config.inshape,
                              self.config.n_classes,
                              self.config['conv_depth'],
                              self.config['batch_norm']):
    elif (self.network_name == 'flat_regress_net'):

      # Update potentially non-standard network parameters
      self.config.set_default('conv_depth',16)
      self.config.set_default('batch_norm',False)
      self.config.set_default('n_layers',8)
      self.config.set_default('conv_pattern',[3])
      self.config.set_default('output_activation','softmax')

      self.model = networks.flat_regress_net(network_config)
      self.model = flex_unet(network_config.inshape,
                             self.config.n_classes,
                             self.config['conv_depth'],
                             self.config['batch_norm'],
                             self.config['n_layers'],
                             self.config['conv_pattern'],
                             self.config['output_activation']):

    else:
      raise NotImplementedError('Unknown network name')


  def calculate_training_memory_usage(batch_size):
      # Shamelessly copied from
      # https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
      # but not tested rigorously
      shapes_mem_count = 0
      for l in self.model.layers:
          single_layer_mem = 1
          for s in l.output_shape:
              if s is None:
                  continue
              single_layer_mem *= s
          shapes_mem_count += single_layer_mem
  
      trainable_count = np.sum([K.count_params(p) for p in set(self.model.trainable_weights)])
      non_trainable_count = np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)])
  
      number_size = 4.0
      if K.floatx() == 'float16':
          number_size = 2.0
      if K.floatx() == 'float64':
          number_size = 8.0
  
      total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
      gbytes = np.round(total_memory / (1024.0 ** 3), 3)
      return gbytes



######## Deprecated.  Let's migrate things upwards as necessary
class NeuralNetwork(object):
    model = None
    history = None
    _verbosity = 1
    _filename_model = 'model.h5'
    _filename_history = 'history.pkl'
    _dir_tensorboard = 'tensorboard'
    _initial_epoch = 0
    # TODO:  break the following out into a config? not critical unless there's a reason for vastly different values
    _cbs_es_min_delta = 10**-4
    _cbs_es_patience = 50
    _cbs_rlr_factor = 0.5
    _cbs_rlr_min_delta = 10**-4
    _cbs_rlr_patience = 10
    _cbs_period = 5
    _cbs_tb_histogram_freq = 0
    _cbs_tb_write_graph = True
    _cbs_tb_write_grads = False
    _cbs_tb_write_images = True
    _cbs_tb_update_freq = 'epoch'
    _callbacks = [
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.EarlyStopping(monitor='loss', min_delta=_cbs_es_min_delta, patience=_cbs_es_patience),
        keras.callbacks.ReduceLROnPlateau(
            factor=_cbs_rlr_factor, monitor='loss', min_delta=_cbs_rlr_min_delta, patience=_cbs_rlr_patience),
    ]
    _generator_max_queue_size = 100
    _generator_workers = 1
    _generator_use_multiprocessing = True

    def __init__(self, model=None, dir_load=None, custom_objects=None):
        assert model is None or dir_load is None, 'Must either pass compiled model or directory with saved model data'
        if model:
            self.model = model
        elif dir_load:
            self.model = keras.models.load_model(
                os.path.join(dir_load, self._filename_model), custom_objects=custom_objects)
            with open(os.path.join(dir_load, self._filename_history), 'rb') as file_:
                self.history = pickle.load(file_)
            self._initial_epoch = len(self.history['lr'])
            K.set_value(self.model.optimizer.lr, self.history['lr'][-1])

    def calculate_model_memory_usage(self, batch_size):
        return calculate_model_memory_usage(self.model, batch_size)

    def fit_sequence(self, train_sequence, dir_out, max_training_epochs, validation_sequence=None):
        # Prep callbacks with dynamic parameters
        filepath_model = os.path.join(dir_out, self._filename_model)
        filepath_history = os.path.join(dir_out, self._filename_history)
        dir_tensorboard = os.path.join(dir_out, self._dir_tensorboard)
        callbacks_dynamic = [
            HistoryCheckpoint(
                filepath_history, existing_history=self.history, period=self._cbs_period, verbose=self._verbosity),
            keras.callbacks.ModelCheckpoint(filepath_model, period=self._cbs_period, verbose=self._verbosity),
            keras.callbacks.TensorBoard(
                dir_tensorboard, histogram_freq=self._cbs_tb_histogram_freq, write_graph=self._cbs_tb_write_graph,
                write_grads=self._cbs_tb_write_grads, write_images=self._cbs_tb_write_images,
                update_freq=self._cbs_tb_update_freq
            ),
        ]
        callbacks = self._callbacks + callbacks_dynamic
        # Train model
        self.model.fit_generator(
            train_sequence, epochs=max_training_epochs, callbacks=callbacks, validation_data=validation_sequence,
            max_queue_size=self._generator_max_queue_size, use_multiprocessing=self._generator_use_multiprocessing,
            workers=self._generator_workers, initial_epoch=self._initial_epoch, verbose=self._verbosity,
        )

    def evaluate(self, evaluate_sequence):
        return self.model.evaluate_generator(
            evaluate_sequence, max_queue_size=self._generator_max_queue_size,
            use_multiprocessing=self._generator_use_multiprocessing, workers=self._generator_workers,
            verbose=self._verbosity
        )

    def predict(self, inputs):
        return self.model.predict(inputs, batch_size=None, verbose=self._verbosity)

    def predict_sequence(self, predict_sequence):
        return self.model.predict_generator(
            predict_sequence, max_queue_size=self._generator_max_queue_size,
            use_multiprocessing=self._generator_use_multiprocessing, workers=self._generator_workers,
            verbose=self._verbosity
        )


class HistoryCheckpoint(keras.callbacks.Callback):

    def __init__(self, filepath, existing_history=None, period=1, verbose=0):
        super().__init__()
        self.filepath = filepath
        if existing_history is None:
            existing_history = dict()
        self.existing_history = existing_history
        self.period = period
        self.verbose = verbose
        self.epochs_since_last_save = 0
        self.epoch_begin = None

    def on_train_begin(self, logs=None):
        for key in ('epoch_start', 'epoch_finish'):
            self.existing_history.setdefault(key, list())
        super().on_train_begin(logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_begin = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Update times
        epoch_end = datetime.datetime.now()
        self.existing_history['epoch_start'].append(self.epoch_begin)
        self.existing_history['epoch_finish'].append(epoch_end)
        self.epoch_begin = None
        # Save if necessary
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.period:
            return
        self.epochs_since_last_save = 0
        history = self._format_history(self.model.history, self.existing_history)
        with open(self.filepath, 'wb') as file_:
            pickle.dump(history, file_)
        if self.verbose > 0:
            print('\nEpoch %05d: saving history to %s' % (epoch + 1, self.filepath))

    def _format_history(self, new_history, old_history=None):
        if old_history is None:
            old_history = dict()
        for key, value in new_history.history.items():
            old_history.setdefault(key, list()).extend(value)
        return old_history

