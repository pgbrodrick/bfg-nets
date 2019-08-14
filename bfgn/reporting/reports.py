import logging
import os
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy import stats
import numpy as np

from bfgn.configuration import configs
from bfgn.data_management import data_core
from bfgn.experiments import experiments
from bfgn.reporting import samples
from bfgn.reporting.visualizations import histories, logs, model_performance, networks, samples as samples_viz


plt.switch_backend('Agg')  # Needed for remote server plotting

_logger = logging.getLogger(__name__)

_FILENAME_MODEL_REPORT = 'model_report.pdf'

_LABEL_CATEGORICAL = 'CATEGORICAL'
_LABEL_CONTINUOUS = 'CONTINUOUS'


class Reporter(object):
    data_container = None
    experiment = None

    def __init__(
            self,
            data_container: data_core.DataContainer,
            experiment: experiments.Experiment,
            config: configs.Config
    ) -> None:
        errors = config.get_human_readable_config_errors(include_sections=['model_reporting'])
        assert not errors, errors
        self.data_container = data_container
        self.experiment = experiment

    def create_model_report(self) -> None:
        filepath_report = os.path.join(self.experiment.config.model_training.dir_out, _FILENAME_MODEL_REPORT)
        with PdfPages(filepath_report) as pdf:
            _logger.info('Plot Summary')
            self._add_figures(self.plot_model_summary(), pdf)
            _logger.info('Plot Training Sequence Figures')
            sampled = samples.Samples(
                self.data_container.training_sequence, self.experiment.model, self.experiment.config,
                self.experiment.is_model_trained, self.data_container.feature_band_types, 
                self.data_container.response_band_types, data_sequence_label='Training'
            )
            self._create_model_report_for_sequence(sampled, pdf)
            _logger.info('Plot Validation Sequence Figures')
            validation_sampled = samples.Samples(
                self.data_container.validation_sequence, self.experiment.model, self.experiment.config,
                self.experiment.is_model_trained, self.data_container.feature_band_types, 
                self.data_container.response_band_types, data_sequence_label='Validation'
            )
            self._create_model_report_for_sequence(validation_sampled, pdf)

            if ('R' in self.data_container.response_band_types):
                self._add_figures(self.plot_regression_deviation(sampled, validation_sampled), pdf)

            _logger.info('Plot Model History')
            self._add_figures(self.plot_model_history(), pdf)
            self._add_figures(self.plot_log_warnings_and_errors(), pdf)

    def _create_model_report_for_sequence(self, sampled: samples.Samples, pdf: PdfPages) -> None:
        if self.experiment.is_model_trained and self._get_response_data_types() is _LABEL_CATEGORICAL:
            self._add_figures(self.plot_classification_report(sampled), pdf)
            self._add_figures(self.plot_confusion_matrix(sampled), pdf, tight=False)
        self._add_figures(self.plot_sample_histograms(sampled), pdf)
        self._add_figures(self.plot_samples(sampled), pdf)
        if self.experiment.config.model_reporting.network_progression_show_full:
            self._add_figures(self.plot_network_feature_progression(sampled, compact=False), pdf)
        if self.experiment.config.model_reporting.network_progression_show_compact:
            self._add_figures(self.plot_network_feature_progression(sampled, compact=True), pdf)
        self._add_figures(self.plot_spatial_error(sampled), pdf)

    def _add_figures(self, figures: List[plt.Figure], pdf: PdfPages, tight: bool = True) -> None:
        for fig in figures:
            pdf.savefig(fig, bbox_inches='tight' if tight else None)

    def plot_confusion_matrix(self, sampled: samples.Samples) -> List[plt.Figure]:
        return model_performance.plot_confusion_matrix(sampled)

    def plot_model_summary(self) -> List[plt.Figure]:
        return networks.plot_model_summary(self.experiment.model)

    def plot_model_history(self) -> List[plt.Figure]:
        return histories.plot_history(self.experiment.history)

    def plot_log_warnings_and_errors(self) -> List[plt.Figure]:
        return logs.plot_log_warnings_and_errors(self.data_container.config, self.experiment.config)

    def plot_network_feature_progression(
            self,
            sampled: samples.Samples,
            compact: bool,
            max_pages: int = None,
            max_filters: int = None
    ) -> List[plt.Figure]:
        return networks.plot_network_feature_progression(
            sampled,
            compact=compact,
            max_pages=max_pages or self.experiment.config.model_reporting.network_progression_max_pages,
            max_filters=max_filters or self.experiment.config.model_reporting.network_progression_max_filters
        )

    def plot_samples(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_samples_per_page: int = None,
            max_features_per_page: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        if self._get_response_data_types() is _LABEL_CATEGORICAL:
            plotter = samples_viz.plot_classification_samples
        elif self._get_response_data_types() is _LABEL_CONTINUOUS:
            plotter = samples_viz.plot_regression_samples
        max_responses_per_page = max_responses_per_page or self.experiment.config.model_reporting.max_responses_per_page
        return plotter(
            sampled,
            max_pages=max_pages or self.experiment.config.model_reporting.max_pages_per_figure,
            max_samples_per_page=max_samples_per_page or self.experiment.config.model_reporting.max_samples_per_page,
            max_features_per_page=max_features_per_page or self.experiment.config.model_reporting.max_features_per_page,
            max_responses_per_page=max_responses_per_page
        )

    def plot_sample_histograms(
            self, sampled: samples.Samples, max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        max_responses_per_page = max_responses_per_page or self.experiment.config.model_reporting.max_responses_per_page
        return samples_viz.plot_sample_histograms(
            sampled,
            max_responses_per_page=max_responses_per_page
        )

    def plot_regression_deviation(
            self, train_sampled: samples.Samples, val_sampled: samples.Samples=None
    ) -> List[plt.Figure]:

        return_figs = []

        for _r in range(len(self.data_container.response_band_types)):
            fig = plt.figure(figsize=(5*(1+int(val_sampled is not None)),4.5))
            gs1 = gridspec.GridSpec(1, 1+ int(val_sampled is not None))
            if (self.data_container.response_band_types[_r] == 'R'):
                bounds = self.parity_plot(train_sampled.raw_predictions,train_sampled.raw_responses,plt.subplot(gs1[0,0]),_r)
                plt.subplot(gs1[0,0]).set_title('Training')
                if (val_sampled is not None):
                    self.parity_plot(val_sampled.raw_predictions,val_sampled.raw_responses,plt.subplot(gs1[0,1]),_r,bounds=bounds)
                    plt.subplot(gs1[0,1]).set_title('Validation')
            fig.suptitle('Response ' + str(_r))
            return_figs.append(fig)
        
        return return_figs

    def parity_plot(self, pred_Y: np.array, test_Y: np.array, ax: plt.Axes, response_ind: int, bins: int=200, bounds=[]):

        loss_window_radius = self.experiment.config.data_build.loss_window_radius
        window_radius = self.experiment.config.data_build.window_radius
        buffer = int(window_radius - loss_window_radius)

        if buffer == 0:
            test_Y = test_Y[...,response_ind].flatten()
            pred_Y = pred_Y[...,response_ind].flatten()
        else:
            test_Y = test_Y[:,buffer:-buffer,buffer:-buffer,response_ind].flatten()
            pred_Y = pred_Y[:,buffer:-buffer,buffer:-buffer,response_ind].flatten()

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.squeeze(test_Y),np.squeeze(pred_Y))
        mad = str(round(np.mean(np.abs(pred_Y-test_Y)),3))
        rmse = str(round(np.sqrt(np.mean(np.power(pred_Y-test_Y,2))),3))
        r2o = str(round(1 - np.sum(np.power(test_Y - pred_Y,2)) / (np.sum(np.power(test_Y - np.mean(pred_Y),2))),3))
        r2 = str(round(r_value**2,3))
        
        if (len(bounds) == 0):
            pmin = np.min(test_Y)
            pmax = np.max(test_Y)
        else:
            pmin = bounds[0]
            pmax = bounds[1]
        z,xrange,yrange = np.histogram2d(test_Y,pred_Y,bins=bins,range=[[pmin, pmax], [pmin, pmax]] )
        ax.patch.set_facecolor('white')
        ax.imshow(np.log(z.T), extent = (pmin,pmax,pmin,pmax),cmap = cm.hot,origin='lower',interpolation='nearest')
        ax.plot([pmin,pmax],[pmin,pmax],color='blue',lw=2,ls='--')
        fs = 8
        ax.text(pmin+(pmax-pmin)*0.05,pmin+(pmax-pmin)*0.95,'MAD: ' + mad,fontsize=fs)
        ax.text(pmin+(pmax-pmin)*0.05,pmin+(pmax-pmin)*0.90,'RMSE: ' + rmse,fontsize=fs)
        ax.text(pmin+(pmax-pmin)*0.05,pmin+(pmax-pmin)*0.85,'R${^2}$${_\mathrm{o}}$: ' + r2o,fontsize=fs)
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        
        ax.set_xlim([pmin,pmax])
        ax.set_ylim([pmin,pmax])

        return [pmin,pmax]


    def plot_spatial_error(
            self,
            sampled: samples.Samples,
            max_pages: int = None,
            max_responses_per_page: int = None
    ) -> List[plt.Figure]:
        if self._get_response_data_types() is _LABEL_CATEGORICAL:
            plotter = model_performance.plot_spatial_classification_error
        elif self._get_response_data_types() is _LABEL_CONTINUOUS:
            plotter = model_performance.plot_spatial_regression_error
        max_responses_per_page = max_responses_per_page or self.experiment.config.model_reporting.max_responses_per_page
        return plotter(
            sampled,
            max_pages=max_pages or self.experiment.config.model_reporting.max_pages_per_figure,
            max_responses_per_page=max_responses_per_page
        )

    def plot_classification_report(self, sampled: samples.Samples) -> List[plt.Figure]:
        return model_performance.plot_classification_report(sampled)

    def _get_response_data_types(self) -> str:
        data_types = set([dt for file_dts in self.experiment.config.raw_files.response_data_type for dt in file_dts])
        if data_types == {'C'}:
            return _LABEL_CATEGORICAL
        elif data_types == {'R'}:
            return _LABEL_CONTINUOUS
        elif data_types == {'C', 'R'}:
            raise AssertionError('Reporter does not currently support mixed response data types.')
        else:
            raise AssertionError('Unexpected data types found: {}.'.format(data_types))
