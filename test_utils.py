import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path, PurePath
import torch
from models import build_model

country_shorthands = {
    'China_Drone': 'CD',
    'China_MotorBike': 'CM',
    'Czech': 'CZ',
    'India': 'IN',
    'Japan': 'JP',
    'Norway': 'NW',
    'United_States': 'US',
    'combined': 'holdout'
}

def plot_logs(logs, fields, ewm_col = 0, log_name = 'log.txt'):
    # set style
    sns.set_style("whitegrid")

    # specify font family
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : 14}
    plt.rc('font', **font)

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 6))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
    #    ax.legend([Path(p).name for p in logs])
        ax.set_title(field)
    return fig, axs

def load_model_from_ckp(ckp_path, args):
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(ckp_path, map_location = 'cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval();
    return model, criterion, postprocessors