import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import networks, rs_data, results
from rsCNN.networks.experiment import Experiment


plt.switch_backend('Agg')  # Needed for remote server plotting


# TODO:  add page with printed warnings and errors from log file, especially following line from NanTermination:
#  Batch 0: Invalid loss, terminating training

class ExperimentReport(object):
    experiment = None

    def __init__(self, experiment: Experiment, test_sequence: BaseSequence):
        self.experiment = experiment
        self.test_sequence = test_sequence

    def create_report(self):
        filepath_report = os.path.join(
            self.experiment.network_config['model']['dir_out'],
            self.experiment.network_config['model']['model_name'] + '.pdf'
        )
        with PdfPages(filepath_report) as pdf:
            # Model summary
            pdf.savefig(networks.print_model_summary(self.experiment.model))
            # Model history
            pdf.savefig(networks.plot_history(self.experiment.history))
            # Input examples and their scaled representations
            for fig in rs_data.plot_raw_and_scaled_input_examples(self.test_sequence):
                pdf.savefig(fig)
            # Output examples and their scaled representations
            for fig in results.plot_predictions(self.test_sequence, self.experiment):
                pdf.savefig(fig)
            # Compact network visualization
            for fig in networks.visualize_feature_progression(self.test_sequence, self.experiment.model):
                pdf.savefig(fig,bbox_inches='tight')
            # Expanded network visualization
            for fig in networks.visualize_feature_progression(self.test_sequence, self.experiment.model, full_vertical=True):
                pdf.savefig(fig,bbox_inches='tight')
            # TODO:  migrate the last reporting functions here


def generate_eval_report(cnn, features, responses, weights, fold_assignments, verification_fold, feature_transform, response_transform, data_config):

    # TODO: migrate these functions above
    # TODO: convert to lined graphs (borrow code from watershed work)
    if (prediction_histogram_comp):
        figs = plot_prediction_histograms(responses.copy(),
                                          weights.copy(),
                                          cnn.predict(feature_transform.transform(features)),
                                          fold_assignments,
                                          verification_fold,
                                          response_transform,
                                          data_config.response_nodata_value)
        for fig in figs:
            pdf.savefig(fig)

    if (spatial_error_concentration):
        figs = spatial_error(responses.copy(),
                             response_transform.inverse_transform(
                                 cnn.predict(feature_transform.transform(features))),
                             weights.copy(),
                             data_config.response_nodata_value)
        for fig in figs:
            pdf.savefig(fig)
    # if (visual_stitching_artifact_check):
    # if (quant_stitching_artificat_check):
