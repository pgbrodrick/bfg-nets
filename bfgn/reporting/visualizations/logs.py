import os
import re
import textwrap
from typing import List

import matplotlib.pyplot as plt

from bfgn.configuration import configs
from bfgn.data_management import data_core
from bfgn.experiments import experiments

plt.switch_backend("Agg")  # Needed for remote server plotting


_LINE_CHARACTER_LIMIT = 140
_LINE_INDENT = "    "


def plot_log_warnings_and_errors(config_data: configs.Config, config_model: configs.Config) -> List[plt.Figure]:
    figures = list()
    filepath_logs_data = data_core.get_log_filepath(config_data)
    figures.append(_plot_log_warnings_and_errors(filepath_logs_data, "Built data"))
    filepath_logs_model = experiments.get_log_filepath(config_model)
    figures.append(_plot_log_warnings_and_errors(filepath_logs_model, "Model training"))
    return figures


def _plot_log_warnings_and_errors(filepath_log: str, log_label: str) -> plt.Figure:
    if os.path.exists(filepath_log):
        with open(filepath_log) as file_:
            raw_lines = [
                re.sub("\n", "", line) for line in file_.readlines() if re.search("(warn|error)", line, re.IGNORECASE)
            ]
        if not raw_lines:
            raw_lines = ["{} log report:  no lines were found containing obvious warnings or errors".format(log_label)]
        else:
            raw_lines.insert(
                0,
                "{} log report:  {} lines were found possibly containing warnings or errors".format(
                    log_label, len(raw_lines)
                ),
            )
    else:
        raw_lines = ["{} log report:  no log file was found at {}".format(log_label, filepath_log)]
    wrapped_lines = list()
    for line in raw_lines:
        wrapped = textwrap.wrap(line, width=_LINE_CHARACTER_LIMIT, subsequent_indent=_LINE_INDENT)
        wrapped_lines.append("\n".join(wrapped))
    finished_lines = "\n\n".join(wrapped_lines)
    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, finished_lines, **{"fontsize": 8, "fontfamily": "monospace"})
    plt.axis("off")
    return fig
