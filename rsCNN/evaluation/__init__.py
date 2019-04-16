import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import networks, rs_data, results
from rsCNN.networks.experiments import Experiment


plt.switch_backend('Agg')  # Needed for remote server plotting


# TODO:  add page with printed warnings and errors from log file, especially following line from NanTermination:
#  Batch 0: Invalid loss, terminating training

class ExperimentReport(object):
    experiment = None

    # TODO: Fabina, I'm trying to get a better handle on your intention with the test_sequence.  Right now, train and validation
    # sequences are already attached to experiment, as of fit.  So is this intended to be a true test sequence (which has never been
    # attached)?  If so, I can adapt these below functions, and they'll make more sense, since right now I'm only doing applicaitons
    # based on the test_sequence, which is screwy.  It should be happening for all three.  Relatively easy change to make, but I didn't
    # realize this was the intent till I was all the way through, and want to make this change when I'm more fresh.
    def __init__(self, experiment: Experiment, train_sequence: BaseSequence, validation_sequence: BaseSequence = None, test_sequence: BaseSequence = None):
        self.experiment = experiment
        self.train_sequence = train_sequence
        self.validation_sequence = validation_sequence
        self.test_sequence = test_sequence

    def create_report(self):
        filepath_report = os.path.join(self.experiment.network_config['model']['dir_out'], 'evaluation_report.pdf')
        with PdfPages(filepath_report) as pdf:
            # Model summary
            pdf.savefig(networks.print_model_summary(self.experiment.model), bbox_inches='tight')
            # Input examples and their scaled representations
            for fig in rs_data.plot_raw_and_scaled_input_examples(self.train_sequence):
                pdf.savefig(fig, bbox_inches='tight')
            # Output examples and their scaled representations
            for fig in results.plot_raw_and_scaled_result_examples(self.train_sequence, self.experiment):
                pdf.savefig(fig, bbox_inches='tight')
            # Compact network visualization
            for fig in networks.visualize_feature_progression(self.train_sequence, self.experiment.model, compact=True):
                pdf.savefig(fig, bbox_inches='tight')
            # Expanded network visualization
            for fig in networks.visualize_feature_progression(self.train_sequence, self.experiment.model):
                pdf.savefig(fig, bbox_inches='tight')

            # TODO: Fabina, going through it more and I'd prefer not to have a separate ResultsReport class.  The only inherrent
            # benefit is not re-performing predictions.  That function is currently cheap, and even if it weren't we could just
            # do it here and pass it in, eliminating the need for a single-use class with one asset

            # Plot Spatial Error
            for fig in results.spatial_error(self.train_sequence, self.experiment):
                pdf.savefig(fig, bbox_inches='tight')

            # TODO: Fabina, I'm not happy with the way these sequences are called at all (see note above).
            # Plot Training Sequence
            for fig in results.single_sequence_prediction_histogram(self.train_sequence, self.experiment, 'Training'):
                pdf.savefig(fig, bbox_inches='tight')
            # Plot Validation Sequence
            if (self.validation_sequence is not None):
                for fig in results.single_sequence_prediction_histogram(self.validation_sequence, self.experiment, 'Validation'):
                    pdf.savefig(fig, bbox_inches='tight')

            # Model history
            pdf.savefig(networks.plot_history(self.experiment.history), bbox_inches='tight')
            # TODO
            # weight_visualization
            # visual_stitching_artifact_check
            # quant_stitching_artificat_check
