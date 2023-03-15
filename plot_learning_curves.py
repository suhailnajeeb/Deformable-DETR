from test_utils import plot_logs
from pathlib import Path
import matplotlib.pyplot as plt

outDir = 'out/iter_2s'
figDir = 'out/iter_2s'

# ensure figdir exists
Path(figDir).mkdir(parents = True, exist_ok = True)

log_directory = [Path(outDir)]

fields_of_interest = (
    'loss',
    'mAP'
)

fig, axs = plot_logs(log_directory, fields_of_interest)

# save figure
fig.savefig(Path(figDir) / 'loss_map.png')
plt.show()