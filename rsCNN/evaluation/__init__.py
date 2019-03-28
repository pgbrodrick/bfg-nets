import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.evaluation import networks, rs_data, results
from rsCNN.networks.model import Experiment


plt.switch_backend('Agg')  # Needed for remote server plotting


class ExperimentReport(object):
    experiment = None

    def __init__(self, experiment: Experiment):
        self.experiment = experiment

    def create_report(self):
        filepath_report = os.path.join(
            self.experiment.network_config['model']['dir_out'],
            self.experiment.network_config['model']['model_name'] + '.pdf'
        )
        with PdfPages(filepath_report) as pdf:
            pdf.savefig(networks.print_model_summary(self.experiment.model))
            pdf.savefig(networks.plot_history(self.experiment.history))
            for fig in rs_data.plot_raw_and_scaled_input_examples(self.experiment.test_sequence):
                pdf.savefig(fig)
            for fig in results.plot_predictions(self.experiment.test_sequence, self.experiment):
                pdf.savefig(fig)
            # TODO:  migrate the last reporting functions here


def generate_eval_report(cnn, features, responses, weights, fold_assignments, verification_fold, feature_transform, response_transform, data_config):

    visual_stitching_artifact_check = True
    quant_stitching_artificat_check = True

    n_examples = 10

    with PdfPages(report_name) as pdf:

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
