import pickle

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_hss_csi(df, thresholds, names, savefig_fname=None):
    x_tick = np.array(thresholds, dtype=np.int16)

    font = {'family': 'sans-serif', 'weight': 'bold', 'size': 14}
    axes = {'titlesize': 18, 'titleweight': 'bold', 'labelsize': 16, 'labelweight': 'bold'}

    mpl.rc('font', **font)  # pass in the font dict as kwargs
    mpl.rc('axes', **axes)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7.5), dpi=100, facecolor='w')
    # ax2 = ax.twinx()
    ax.set_zorder(1)  # default zorder is 0 for ax and ax2，數字大的在上面
    ax.patch.set_visible(False)  # prevents ax from hiding ax2 # ax.set_frame_on(False) for new version

    for i in range(len(names)):
        pt, = ax.plot(range(len(thresholds) - 1), df.loc[names[i]], '-o', lw=5, label=names[i], markersize=10,
                      zorder=3)  # zorder只能用在同一個ax中

    ax.grid(axis='y', ls='--')
    ax.set_axisbelow(True)
    ax.set_xlim([-0.5, 6])
    ax.set_ylim([0, 0.8])
    ax.set_xticklabels(x_tick)
    # set legend order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0, 1, 2]
    lg = ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        ncol=3,
        frameon=True,  #邊框
        fancybox=False,  #是否圓邊
        edgecolor='black',
        bbox_to_anchor=(0, -0.3, 1, 0.2),  # [x0, y0, width, height]
        loc='upper left',  # 這邊的upper left是指(x0,y0)座標對應到legend的哪個角落
        mode='expand',  # expand才會讓圖例展開
        handlelength=5,
    )
    plt.tight_layout()  # 使子圖合適地跟圖形匹配
    if savefig_fname is not None:
        plt.savefig(savefig_fname, dpi=300, format='png', bbox_extra_artists=(lg, ), bbox_inches='tight')


def _plot_prediction_with_target_simple(output, target, ax, row_idx, data_idx):
    import seaborn as sns
    assert output.shape[0] == 3
    assert target.shape[0] == 3
    for col_idx in range(3):
        output_t = output[col_idx, ...]
        target_t = target[col_idx, ...]

        sns.heatmap(output_t, ax=ax[row_idx, 2 * col_idx], vmax=50)
        sns.heatmap(target_t, ax=ax[row_idx, 2 * col_idx + 1], vmax=50)
        ax[row_idx, 2 * col_idx].set_title(f'Pred: {data_idx}-{col_idx}', )
        ax[row_idx, 2 * col_idx + 1].set_title(f'Tar: {data_idx}-{col_idx}')

        ax[row_idx, 2 * col_idx].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx, 2 * col_idx].axis('off')
        ax[row_idx, 2 * col_idx + 1].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx, 2 * col_idx + 1].axis('off')


def _plot_prediction_with_target_continuous(output, target, ax, col_idx, ts):
    import seaborn as sns
    assert output.shape[0] == 3
    assert target.shape[0] == 3
    ts = ts.strftime('%Y%m%d-%H%M')

    target_t = target[0, ...]
    sns.heatmap(target_t, ax=ax[0, col_idx], vmax=50)
    ax[0, col_idx].set_title(f'{ts}')
    ax[0, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
    ax[0, col_idx].axis('off')

    for row_idx in range(3):
        output_t = output[row_idx, ...]
        sns.heatmap(output_t, ax=ax[row_idx + 1, col_idx], vmax=50)

        # ax[row_idx + 1, col_idx].set_title(f'P: {col_idx}', )
        ax[row_idx + 1, col_idx].tick_params(left=False, right=False, top=False, bottom=False)
        ax[row_idx + 1, col_idx].axis('off')


def plot_prediction_with_target(model,
                                data_loader,
                                count,
                                with_prior=False,
                                img_sz=3,
                                downsample_factor=1,
                                continuous_plot=False,
                                post_processing_fn=None,
                                index_list=None):
    import matplotlib.pyplot as plt
    if continuous_plot:
        _, ax = plt.subplots(figsize=(img_sz * count, img_sz * 4), ncols=count, nrows=4)
    else:
        _, ax = plt.subplots(figsize=(img_sz * 6, img_sz * count), ncols=6, nrows=count)

    if index_list is None:
        index_list = np.random.choice(np.arange(len(data_loader)), size=count, replace=False)
    print(index_list)
    with torch.no_grad():
        for i, idx in tqdm(enumerate(index_list)):
            data = data_loader[idx]
            data = [torch.Tensor(elem).cuda() for elem in data]
            inp, target, mask = data[:3]
            prior_inp = [elem[None, ...] for elem in data[3:]]
            output = model(inp[None, ...])
            if post_processing_fn is not None:
                output = post_processing_fn(output)

            if with_prior:
                prior = model.module._prior_model(*prior_inp)
                prior = prior.permute(1, 0, 2, 3)
                output = output * prior
            output = output.cpu().numpy()[:, 0, ...]
            target = target.cpu().numpy()
            if downsample_factor > 1:
                output = output[:, ::downsample_factor, ::downsample_factor]
                target = target[:, ::downsample_factor, ::downsample_factor]
            if continuous_plot:
                _plot_prediction_with_target_continuous(output, target, ax, i, data_loader.target_ts(idx))
            else:
                _plot_prediction_with_target_simple(output, target, ax, i, idx)


def plot_prediction_around_ts(
    model,
    data_loader,
    ts,
    window=4,
    with_prior=False,
    downsample_factor=5,
    img_sz=2,
):
    assert with_prior == False
    assert data_loader._sampling_rate == 1

    index = data_loader.get_index_from_target_ts(ts)
    start_index = max(0, index - window)
    end_index = min(len(data_loader), index + window + 1)
    index_list = list(range(start_index, end_index))
    plot_prediction_with_target(
        model,
        data_loader,
        len(index_list),
        with_prior=with_prior,
        img_sz=img_sz,
        downsample_factor=downsample_factor,
        index_list=index_list,
        continuous_plot=True,
    )
