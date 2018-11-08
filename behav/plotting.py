from __future__ import absolute_import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from . import utils
import scipy as sp
from scipy import ndimage
from six.moves import zip


def plot_stars(p, x, y, size='large', horizontalalignment='center', **kwargs):
    ''' Plots significance stars '''
    plt.text(x, y, utils.stars(p), size=size, horizontalalignment=horizontalalignment, **kwargs)


def plot_linestar(p, x1, x2, y):
    hlines(y, x1, x2)
    plot_stars(0.5 * (x1 + x2), y + 0.02, utils.stars(p), size='large', horizontalalignment='center')


def _date_labels(dates):
    months = np.array([mdt.month for mdt in dates])
    months_idx = np.append([True], months[:-1] != months[1:])
    strfs = np.array(['%d', '%b %d'])[months_idx.astype(int)]
    return [dt.strftime(strf) for dt, strf in zip(dates, strfs)]


def plot_filtered_performance_calendar(subj, df, num_days=7, **kwargs):
    '''
    plots a calendar view of the performance for a subject on the past num_days.
    '''
    df2 = utils.filter_normal_trials(utils.filter_recent_days(df, num_days))
    return plot_performance_calendar(subj, df2, **kwargs)


def plot_performance_calendar(subj, data_to_analyze, disp_counts=False, vmins=(0, 0, 0), vmaxs=(None, 1, None)):
    '''
    plots a calendar view of performance for a subject.
    Plots all trials from data_to_analyze so make sure it is filtered.

    Parameters:
    -----------
    subj : str
        the subject
    data_to_analyze : pandas DataFrame
        filtered data to plot. Can be a slice, a copy is made anyways.
    disp_counts : boolean
        display values in grid, removes colorbars, default False
    vmins, vmaxs : iterable of floats, length 3, optional
        Values to anchor the colormaps. If None, they are inferred from the data.
    '''
    data_to_analyze = data_to_analyze.copy()
    data_to_analyze['date'] = data_to_analyze.index.date
    data_to_analyze['hour'] = data_to_analyze.index.hour

    blocked = data_to_analyze.groupby(['date', 'hour'])
    aggregated = pd.DataFrame(blocked.agg({'correct': lambda x: np.mean(x.astype(float)),
                                           'reward': lambda x: np.sum(x.astype(float)),
                                           'type_': lambda x: np.sum((x == 'normal').astype(float))}).to_records())

    f, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16.0, 4.0))

    columns = ('type_', 'correct', 'reward')
    titles = (subj + ': Trials per hour', 'Accuracy', 'Feeds')
    cmaps = [plt.get_cmap(cmap) for cmap in ('Oranges', 'RdYlBu', 'BuGn')]
    for cmap in cmaps:
        cmap.set_bad(color='Grey')

    pivoted = aggregated.pivot('hour', 'date')

    for i, (column, title, cmap, vmin, vmax) in enumerate(zip(columns, titles, cmaps, vmins, vmaxs)):
        g = sns.heatmap(pivoted[column], annot=disp_counts, ax=ax[i],
                        cmap=cmap, cbar=not disp_counts,
                        vmin=vmin, vmax=vmax)
        g.set_title(title)
    idk_what_ppl_were_thinking = [x[1] for x in list(pivoted.keys())]
    g.set_xticklabels(_date_labels(idk_what_ppl_were_thinking))
    return f


def plot_filtered_accperstim(title, df, num_days=7, **kwargs):
    '''
    plots accuracy per stim for a subject on the past num_days.
    '''
    return plot_accperstim(title, utils.filter_normal_trials(utils.filter_recent_days(df, num_days)), **kwargs)


def plot_accperstim(title, data_to_analyze, stim_ids='stimulus', stims_all=None, label_count_cutoff=50, extract_stim_names=True):
    '''
    percent correct broken out by stimulus and day.

    Parameters:
    -----------
    title : str
        the plot title
    data_to_analyze : pandas DataFrame
        filtered data to plot. Can be a slice, a copy is made anyways.
    stim_ids : str
        label of the column to group-by.
    stims_all : None or list-like
        order of stims. must match values in stim_ids
    label_count_cutoff : int
        max number of stimuli labels. If below this value will sort stim_ids by class.
    extract_stim_names : boolean
        whether to extract stimuli names from full stimuli paths. If true, ignores stim_ids.
    '''
    data_to_analyze = data_to_analyze.copy()
    if extract_stim_names:
        stim_ids = 'stim_name'
        utils.extract_filename(data_to_analyze, target=stim_ids)
    data_to_analyze['date'] = data_to_analyze.index.date

    blocked = data_to_analyze.groupby(['date', stim_ids])
    aggregated = pd.DataFrame(blocked.agg(
        {'correct': lambda x: np.mean(x.astype(float))}).to_records())
    pivoted = aggregated.pivot(stim_ids, 'date', 'correct')
    if stims_all:
        yticklabels = stims_all
    elif len(pivoted) < label_count_cutoff:
        yticklabels = data_to_analyze.groupby(
            ['class_', stim_ids]).index.unique().index.get_level_values(stim_ids).values
    else:
        yticklabels = int(len(pivoted) / label_count_cutoff)
    cmap = sns.diverging_palette(15, 250, as_cmap=True)
    cmap.set_bad(color='k', alpha=0.5)
    plt.figure()
    g = sns.heatmap(pivoted, vmin=0, vmax=1, cmap=cmap,
                    xticklabels=_date_labels(list(pivoted.keys()).values),
                    yticklabels=yticklabels)
    g.set_title(title)
    return g


def plot_daily_accuracy(subj, df, x_axis='trial_num', smoothing='gaussian', day_lim=0):
    '''
    plots the accuracy of the subject throughout the day.
    a preset for the more general plot_accuracy_bias

    Parameters:
    -----------
    subj : str
        the subject
    df : pandas DataFrame
        data frame of behavior data
    x_axis : str
        whether to plot 'time' or 'trial_num' along the x axis
    smoothing : str
        whether to smooth using 'exponential', 'rolling' average,
        'gaussian' filter'
    day_lim : None or non-negative int
        max number of days of trials to include. Zero means just today.
    '''
    return plot_accuracy_bias(subj, df, x_axis=x_axis, smoothing=smoothing, trial_lim=None, day_lim=day_lim,
                              plt_correct_smoothed=True, plt_correct_shade=True, plt_correct_line=True,
                              plt_L_response_smoothed=False, plt_L_response_shade=False, plt_L_response_line=False,
                              plt_R_response_smoothed=False, plt_R_response_shade=False, plt_R_response_line=False,
                              plt_ci=False, block_size=100)


def plot_ci_accuracy(subj, df, x_axis='time', day_lim=7, trial_lim=None, bias=True):
    '''
    plots the accuracy (and bias) of the subject throughout the day.
    a preset for the more general plot_accuracy_bias

    Parameters:
    -----------
    subj : str
        the subject
    df : pandas DataFrame
        data frame of behavior data
    x_axis : str
        whether to plot 'time' or 'trial_num' along the x axis
    trial_lim : None or int
        max number of most recent trials to include
    day_lim : None or non-negative int
        max number of days of trials to include. Zero means just today.
    bias : boolean
        whether to plot the line for the left bias
    '''
    return plot_accuracy_bias(subj, df, x_axis=x_axis, smoothing='rolling', trial_lim=None, day_lim=day_lim,
                              plt_correct_smoothed=True, plt_correct_shade=False, plt_correct_line=False,
                              plt_L_response_smoothed=bias, plt_L_response_shade=False, plt_L_response_line=False,
                              plt_R_response_smoothed=False, plt_R_response_shade=False, plt_R_response_line=False,
                              plt_ci=True, block_size=100)


def plot_accuracy_bias(subj, df, x_axis='time', smoothing='exponential', trial_lim=None, day_lim=7,
                       plt_correct_smoothed=True, plt_correct_shade=True, plt_correct_line=True,
                       plt_L_response_smoothed=False, plt_L_response_shade=False, plt_L_response_line=False,
                       plt_R_response_smoothed=False, plt_R_response_shade=False, plt_R_response_line=False,
                       plt_ci=False, block_size=100):
    '''
    plots the accuracy or bias of the subject.

    Parameters:
    -----------
    subj : str
        the subject
    df : pandas DataFrame
        data frame of behavior data
    x_axis : str
        whether to plot 'time' or 'trial_num' along the x axis
    smoothing : str
        whether to smooth using 'exponential', 'rolling' average,
        'gaussian' filter'
    trial_lim : None or int
        max number of most recent trials to include
    day_lim : None or non-negative int
        max number of days of trials to include. Zero means just today.
    plt_{correct, L_response, R_response}_smoothed : boolean
        whether to plot a smoothed line for the value
    plt_{correct, L_response, R_response}_shade : boolean
        whether to plot a red shaded region filling in the line of actual responses
    plt_{correct, L_response, R_response}_line : boolean
        whether to plot a red line of the actual responses
    '''
    fig = plt.figure(figsize=(16, 2))
    if trial_lim is not None:
        df = df[-trial_lim:]
    if day_lim is not None:
        df = utils.filter_recent_days(df, day_lim)
    df = utils.filter_normal_trials(df)
    if x_axis == 'time':
        x = df.index._mpl_repr()
        use_index = True
    elif x_axis == 'trial_num':
        x = np.arange(len(df))
        use_index = False
    else:
        raise Exception('invalid value for x_axis')

    datas = (df['correct'].astype(float), df['response'] == 'L', df['response'] == 'R')
    plot_smoothed_mask = (plt_correct_smoothed, plt_L_response_smoothed, plt_R_response_smoothed)
    plot_shaded_mask = (plt_correct_shade, plt_L_response_shade, plt_R_response_shade)
    plot_line_mask = (plt_correct_line, plt_L_response_line, plt_R_response_line)

    for data, smoothed, shaded, line in zip(datas, plot_smoothed_mask, plot_shaded_mask, plot_line_mask):

        if shaded:
            plt.fill_between(x, .5, data.values.astype(bool), color='r', alpha=.25)
        if line:
            g = data.plot(color='r', marker='o', linewidth=.5, use_index=use_index)
        if smoothed:
            if smoothing == 'exponential':
                data.ewm(halflife=20).mean().plot(use_index=use_index)
            elif smoothing == 'gaussian':
                plt.plot(x, ndimage.filters.gaussian_filter(
                    data.values.astype('float32'), 3, order=0))
            elif smoothing == 'rolling':
                data.rolling(window=block_size, center=True).mean().plot(use_index=use_index)
            else:
                raise Exception('invalid value for smoothing')

    if plt_ci and smoothing == 'rolling':
        ci = utils.binomial_ci(0.5 * block_size, block_size)
        plt.axhspan(ci[0], ci[1], color='grey', alpha=0.5)
    plt.axhline(y=.5, c='black', linestyle='dotted')
    plt.title('Today\'s Performance: ' + subj)
    plt.xlabel(x_axis)
    return fig


def plot_trial_feeds(behav_data, num_days=7):
    '''
    plots numer of trials and number of feeds for all birds across time

    Parameters:
    -----------
    behav_data : dict of pandas dataframes
        from loading.load_data_pandas
    num_days : non-negative int
        number of days to include data for
    '''
    colors = sns.hls_palette(len(behav_data))
    fig = plt.figure(figsize=(16.0, 4.0))
    ax1 = fig.gca()
    ax2 = ax1.twinx()

    for (subj, df), color in zip(list(behav_data.items()), colors):
        data_to_analyze = utils.filter_recent_days(df, num_days).copy()
        if not data_to_analyze.empty:
            data_to_analyze['date'] = data_to_analyze.index.date
            blocked = data_to_analyze.groupby('date')

            days = np.sort(list(blocked.groups.keys()))
            trials_per_day = blocked['response'].count().values
            line = ax1.plot(days, trials_per_day, label=subj + ' trials per day', c=color)
            if len(days) == 1:
                plot(0, trials_per_day[-1], 'o', c=color, ax=ax1)

            aggregated = blocked.agg({'reward': lambda x: np.sum((x == True).astype(float))})
            aggregated['reward'].plot(ax=ax2, label=subj + ' feeds per day', ls='--', c=color)
            if len(days) == 1:
                ax2.plot(0, aggregated['reward'][0], 'o', c=color)

    plt.title('trials and feeds per day')
    for ax, label, loc in zip((ax1, ax2), ('trials per day', 'feeds per day'), ('upper left', 'upper right')):
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.legend(loc=loc)
    ax1.set_xticklabels(_date_labels(days))
    return fig
