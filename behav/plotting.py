import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils
import scipy as sp
from scipy import ndimage

def plot_stars(p,x,y,size='large',horizontalalignment='center',**kwargs):
    ''' Plots significance stars '''
    plt.text(x,y,stars(p),size=size,horizontalalignment=horizontalalignment,**kwargs)

def plot_linestar(p,x1,x2,y):
    hlines(y, x1, x2)
    plot_stars(0.5*(x1+x2),y+0.02,stars(p),size='large',horizontalalignment='center')

def _date_labels(dates):
    months = np.array([mdt.month for mdt in dates])
    months_idx = np.append([True] , months[:-1] != months[1:])
    strfs = np.array(['%d', '%b %d'])[months_idx.astype(int)]
    return [dt.strftime(strf) for dt, strf in zip(dates, strfs)]

def plot_filtered_performance_calendar(subj,df,num_days=7, **kwargs):
    '''
    plots a calendar view of the performance for a subject on the past num_days.
    '''
    plot_performance_calendar(subj, utils.filter_normal_trials(utils.filter_recent_days(df, num_days)), **kwargs)

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
                                              'type_': lambda x: np.sum((x=='normal').astype(float))}).to_records())

    f, ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(16.0, 4.0))

    columns = ('type_', 'correct', 'reward')
    titles = (subj+': Trials per hour', 'Accuracy', 'Feeds')
    cmaps = [plt.get_cmap(cmap) for cmap in ('Oranges', 'RdYlBu', 'BuGn')]
    for cmap in cmaps:
        cmap.set_bad(color='Grey')
    
    pivoted = aggregated.pivot('hour', 'date')
    
    for i, (column, title, cmap, vmin, vmax) in enumerate(zip(columns, titles, cmaps, vmins, vmaxs)):
        g = sns.heatmap(pivoted[column], annot=disp_counts, ax=ax[i], 
                        cmap=cmap, cbar=not disp_counts,
                        vmin=vmin, vmax=vmax)
        g.set_title(title)
    g.set_xticklabels(_date_labels(pivoted.keys().levels[1]));


def plot_filtered_accperstim(title,df,num_days=7, **kwargs):
    '''
    plots accuracy per stim for a subject on the past num_days.
    '''
    plot_accperstim(title, utils.filter_normal_trials(utils.filter_recent_days(df, num_days)), **kwargs)


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
    aggregated = pd.DataFrame(blocked.agg({'correct': lambda x: np.mean(x.astype(float))}).to_records())
    pivoted = aggregated.pivot(stim_ids, 'date', 'correct')
    if stims_all:
        yticklabels = stims_all
    elif len(pivoted)<label_count_cutoff:
        yticklabels = data_to_analyze.groupby(['class_', stim_ids]).index.unique().index.get_level_values(stim_ids).values
    else:
        yticklabels = int(len(pivoted)/label_count_cutoff)
    cmap = sns.diverging_palette(15, 250, as_cmap=True)
    cmap.set_bad(color = 'k', alpha = 0.5)
    plt.figure()
    g = sns.heatmap(pivoted, vmin=0, vmax=1, cmap=cmap, 
                    xticklabels=_date_labels(pivoted.keys().values),
                    yticklabels=yticklabels)
    g.set_title(title)

def plot_daily_accuracy(subj, df, x_axis='time', smoothing=None):
    '''
    plots the accuracy of the subject throughout the day.
    
    Parameters:
    -----------
    subj : str
        the subject
    df : pandas DataFrame
        data frame of behavior data
    x_axis : str
        whether to plot 'time' or 'trial_num' along the x axis
    smoothing : str or None
        whether to smooth using 'exponential', 'rolling' average, 
        'gaussian' filter', 'none', or None to pick exponential or gaussian
    '''
    fig = plt.figure(figsize=(16, 2)) 
    filtered_data = utils.filter_normal_trials(utils.filter_recent_days(df, 0))
    if x_axis == 'time':
        x = filtered_data.index._mpl_repr()
        use_index=True
        if not smoothing:
            smoothing = 'exponential'
    elif x_axis == 'trial_num':
        x = np.arange(len(filtered_data))
        use_index=False
        if not smoothing:
            smoothing = 'gaussian'
    else:
        raise Exception('invalid value for x_axis')
    
    g = filtered_data['correct'].plot(color='r', marker='o', use_index=use_index)
    plt.fill_between(x, .5, filtered_data['correct'].values.astype(bool), color = 'r', alpha = .25)
    if smoothing == 'exponential':
        filtered_data['correct'].ewm(halflife=20).mean().plot(use_index=use_index, linewidth = 3)
    elif smoothing == 'rolling':
        filtered_data['correct'].rolling(window=10, center=True).mean().plot(color='k', linewidth = 3, use_index=use_index)
    elif smoothing == 'gaussian':
        plt.plot(x, sp.ndimage.filters.gaussian_filter(filtered_data['correct'].values.astype('float32'), 3, order=0), linewidth = 3)
    elif smoothing != 'none':
        raise Exception('invalid value for smoothing')
    
    plt.axhline(y=.5, c='black', linestyle='dotted')
    plt.title('Today\'s Performance: '+subj)
    plt.xlabel(x_axis)
    return