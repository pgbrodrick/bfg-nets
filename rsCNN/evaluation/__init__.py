import os

import keras
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from rsCNN.data_management.sequences import BaseSequence
from rsCNN.evaluation import histories, networks, rs_data, results
from rsCNN.networks import experiments


plt.switch_backend('Agg')  # Needed for remote server plotting


# TODO:  add page with printed warnings and errors from log file, especially following line from NanTermination:
#  Batch 0: Invalid loss, terminating training

def create_report_from_experiment(experiment: experiments.Experiment):
    return create_report(
        experiment.model, experiment.train_sequence, experiment.validation_sequence, experiment.test_sequence,
        experiment.network_config, experiment.history
    )


def create_report(
        model: keras.Model,
        network_config: dict,
        train_sequence: BaseSequence,
        validation_sequence: BaseSequence = None,
        test_sequence: BaseSequence = None,
        history: dict = None,
) -> None:
    filepath_report = os.path.join(network_config['model']['dir_out'], 'evaluation_report.pdf')
    with PdfPages(filepath_report) as pdf:
        # Model summary
        pdf.savefig(networks.print_model_summary(model), bbox_inches='tight')
        # Input examples and their scaled representations
        for fig in rs_data.plot_raw_and_scaled_input_examples(train_sequence, model):
            pdf.savefig(fig, bbox_inches='tight')
        # Output examples and their scaled representations
        for fig in results.plot_raw_and_scaled_result_examples(model, network_config, train_sequence):
            pdf.savefig(fig, bbox_inches='tight')
        # Compact network visualization
        for fig in networks.visualize_feature_progression(train_sequence, model, compact=True):
            pdf.savefig(fig, bbox_inches='tight')
        # Expanded network visualization
        for fig in networks.visualize_feature_progression(train_sequence, model):
            pdf.savefig(fig, bbox_inches='tight')

        # Plot Spatial Error
        for fig in results.spatial_error(model, train_sequence):
            pdf.savefig(fig, bbox_inches='tight')

        # Plot Training Sequence
        for fig in results.single_sequence_prediction_histogram(model, train_sequence, 'Training'):
            pdf.savefig(fig, bbox_inches='tight')
        # Plot Validation Sequence
        if (validation_sequence is not None):
            for fig in results.single_sequence_prediction_histogram(model, validation_sequence, 'Validation'):
                pdf.savefig(fig, bbox_inches='tight')

        # Model history
        pdf.savefig(history.plot_history(history), bbox_inches='tight')
        # TODO
        # weight_visualization
        # visual_stitching_artifact_check
        # quant_stitching_artificat_check
