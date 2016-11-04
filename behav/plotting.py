import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import utils

def plot_stars(p,x,y,size='large',horizontalalignment='center',**kwargs):
    ''' Plots significance stars '''
    plt.text(x,y,stars(p),size=size,horizontalalignment=horizontalalignment,**kwargs)

def plot_linestar(p,x1,x2,y):
    hlines(y, x1, x2)
    plot_stars(0.5*(x1+x2),y+0.02,stars(p),size='large',horizontalalignment='center')

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

    for i, (column, title, cmap, vmin, vmax) in enumerate(zip(columns, titles, cmaps, vmins, vmaxs)):
        pivoted = aggregated.pivot('hour', 'date', column)
        g = sns.heatmap(pivoted, annot=disp_counts, ax=ax[i], 
                        cmap=cmap, cbar=not disp_counts,
                        vmin=vmin, vmax=vmax)
        g.set_title(title)

    months = np.array([mdt.month for mdt in pivoted.keys().values])
    months_idx = np.append([True] , months[:-1] != months[1:])
    strfs = np.array(['%d', '%b %d'])[months_idx.astype(int)]
    g.set_xticklabels([dt.strftime(strf) for dt, strf in zip(pivoted.keys().values, strfs)]);