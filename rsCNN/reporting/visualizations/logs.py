import re
from typing import List

import matplotlib.pyplot as plt


plt.switch_backend('Agg')  # Needed for remote server plotting


def plot_log_warnings_and_errors(path_log: str) -> List[plt.Figure]:
    with open(path_log) as file_:
        lines = [line for line in file_.readlines() if re.search('(warn|error)', line, re.IGNORECASE)]
    lines.insert(0, 'Log report:  lines possibly containing warnings or errors')

    fig, axes = plt.subplots(figsize=(8.5, 11), nrows=1, ncols=1)
    plt.text(0, 0, lines, **{'fontsize': 8, 'fontfamily': 'monospace'})
    plt.axis('off')

    return [fig]


