import os
import re
from typing import List

import matplotlib.pyplot as plt

from rsCNN.configuration import configs
from rsCNN.data_management import data_core
from rsCNN.experiments import experiments


plt.switch_backend('Agg')  # Needed for remote server plotting


def plot_log_warnings_and_errors(config: configs.Config) -> List[plt.Figure]:
    figures = list()
    filepath_logs_data = data_core.get_log_filepath(config)
    figures.append(_plot_log_warnings_and_errors(filepath_logs_data, 'Built data'))
    filepath_logs_model = experiments.get_log_filepath(config)
    figures.append(_plot_log_warnings_and_errors(filepath_logs_model, 'Model training'))
    return figures


def _plot_log_warnings_and_errors(filepath_log: str, log_label: str) -> plt.Figure:
    if os.path.exists(filepath_log):
        with open(filepath_log) as file_:
            lines = [line for line in file_.readlines() if re.search('(warn|error)', line, re.IGNORECASE)]
        if not lines:
            lines = ['{} log report:  no lines were found containing obvious warnings or errors'.format(log_label)]
        else:
            lines.insert(0, '{} log report:  {} lines were found possibly containing warnings or errors'.format(
                log_label, len(lines)))
    else:
        lines = ['{} log report:  no log file was found at {}'.format(log_label, filepath_log)]
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, lines, **{'fontsize': 8, 'fontfamily': 'monospace'})
    plt.axis('off')
    return fig
