import glob
import pandas as pd
import datetime as dt
import os
import numpy as np
import warnings

def load_data_pandas(subjects, data_folder, force_boolean=['reward']):
    '''
    a function that loads data files for a number of subjects into panda DataFrames.
    supports pyoperant files, rDAT files (from c operant behavior scripts), and AllTrials files

    Parameters:
    -----------
    subjects :  list or tuple of str
        bird ids of any length i.e. ('B999', 'B9999')
    data_folder : str
        top level folder for the data containing folders matching the elements of subjects
    force_boolean : list of str, optional
        data columns which will be cast as bool

    Returns:
    --------
    dict
        each key in the dict is a subject string. each value is a pandas dataframe
    '''
    behav_data = {}
    for subj in subjects:
        df_set = []

        # if vogel/pyoperant
        data_files = glob.glob(os.path.join(data_folder,subj,subj+'_trialdata_*.csv'))
        if data_files:
            for data_f in data_files:
                with open(data_f,'rb') as f:
                    try:
                        df = pd.read_csv(f,index_col=['time'],parse_dates=['time'])
                        df = df[~pd.isnull(df.index)]
                        df['data_file'] = data_f
                        df_set.append(df)
                    except ValueError:
                        df = None

        # if ndege/c operant
        data_files = glob.glob(os.path.join(data_folder,subj,subj[1:]+'_match2sample*.2ac_rDAT'))
        if data_files:
            fmt = [('session','i4'),
                   ('trial_number','i4'),
                   ('old_type','b'),
                   ('stimulus','a64'),
                   ('old_class','i4'),
                   ('old_response','i4'),
                   ('old_correct','i4'),
                   ('rt','f4'),
                   ('reinforcement','b'),
                   ('TimeOfDay','a8'),
                   ('old_date','a8'),
                   ];
            for data_f in data_files:
                nheaderrows = 5
                dat = load_rDAT(data_f, nheaderrows=nheaderrows, fmt=fmt)

                year = _read_year_rDAT(data_f, nheaderrows)
                df = pd.DataFrame(dat, columns=zip(*fmt)[0])
                dt_maker = _make_dt_maker(year)
                df['date'] = df.apply(dt_maker, axis=1)
                df.set_index('date', inplace=True)
                df['type_'] = df['old_type'].map(lambda x: ['correction','normal'][x])
                df['response'] = df['old_response'].map(lambda x: ['none', 'L', 'R'][x])
                df['correct'] = df['old_correct'].map(lambda x: [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda x: x['reinforcement'] == 1 and x['correct'] == True, axis=1)
                df['class_'] = df['old_class'].map(lambda x: ['none', 'L', 'R'][x])
                df['data_file'] = data_f
                df_set.append(df)

        # if ndege/c GONOGO operant
        data_files = glob.glob(os.path.join(data_folder,subj,subj[1:]+'*.gonogo_rDAT'))
        if data_files:
            fmt = [('session','i4'),
                   ('trial_number','i4'),
                   ('old_type','b'),
                   ('stimulus','a64'),
                   ('old_class','i4'),
                   ('old_response','i4'),
                   ('old_correct','i4'),
                   ('rt','f4'),
                   ('reinforcement','b'),
                   ('TimeOfDay','a8'),
                   ('old_date','a8'),
                   ];
            for data_f in data_files:
                nheaderrows = 5
                dat = load_rDAT(data_f, nheaderrows=nheaderrows, fmt=fmt)

                year = _read_year_rDAT(data_f, nheaderrows)
                df = pd.DataFrame(dat, columns=zip(*fmt)[0])
                dt_maker = _make_dt_maker(year)
                df['date'] = df.apply(dt_maker, axis=1)
                df.set_index('date', inplace=True)
                df['type_'] = df['old_type'].map(lambda x: ['correction','normal'][x])
                df['response'] = df['old_response'].map(lambda x: ['none', 'C'][x])
                df['correct'] = df['old_correct'].map(lambda x: [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda x: x['reinforcement'] == 1 and x['correct'] == True, axis=1)
                df['class_'] = df['old_class'].map(lambda x: ['none', 'GO', 'NOGO'][x])
                df['data_file'] = data_f
                df_set.append(df)

        # if AllTrials file from probe-the-broab
        data_files = glob.glob(os.path.join(data_folder,subj,subj+'.AllTrials'))
        if data_files:
            col_map = {'StimName': 'stimulus',
                       'Epoch': 'session',
                       'StimulusFile': 'block_name',
                       }
            def _parse(datestr, timestr):
                return dt.datetime.strptime(datestr+timestr,"%Y:%m:%d%H:%M:%S")

            for data_f in data_files:
                nheaderrows = 1
                # try:
                df = pd.read_csv(data_f,
                                 parse_dates={'date':['Date','Time']},
                                 date_parser=_parse,
                                 )
                df.rename(columns=col_map, inplace=True)
                df.set_index('date',inplace=True)
                df['type_'] = df['Correction'].map(lambda x: {0:'normal',1:'correction',243:'error',-1:None}[x])
                df['correct'] = df['ResponseAccuracy'].map(lambda x: [False, True, float('nan')][x])
                df['reward'] = df.apply(lambda x: x['Reinforced'] == 1 and x['correct'] == True, axis=1)
                df['punish'] = df.apply(lambda x: x['Reinforced'] == 1 and x['correct'] == False, axis=1)
                df['class_'] = df['StimClass'].map(lambda x: {0:'none',1:'L',2:'R',243:'error',-1:None}[x])
                df['response'] = df['ResponseSelection'].map(lambda x: ['none', 'L', 'R'][x])
                df['data_file'] = data_f

                is_behave = df['BehavioralRecording'] > 0
                df = df[is_behave]

                df_set.append(df)
                (force_boolean.append(x) for x in ['NeuralRecording','BehavioralRecording'])

                # except ValueError:
                #     df = None
        if df_set:
            #return df_set
            # sort out non-timestamp indexes
            def _validate_time(date_text, date_format = "%Y-%m-%d %H:%M:%S.%f"):
                """ Remove any invalid datetime index"""
                try:
                    return dt.datetime.strptime(date_text, date_format)
                except:
                    return False
            # test for dfs where the index is not datetime
            broken_dfs = np.where([(type(i.index) != pd.core.indexes.datetimes.DatetimeIndex) &(len(i)>0) for i in df_set])[0]

            if len(broken_dfs)> 0:
                warnings.warn('Warning: ' + str(len(broken_dfs))+' Pandas dataframe contained non-datetime indexes')
                for broken_df in broken_dfs:
                    df_set[broken_df].index = [_validate_time(i,"%Y-%m-%d %H:%M:%S.%f") for i in df_set[broken_df].index]
                    df_set[broken_df] = df_set[broken_df][df_set[broken_df].index != False]
                    df_set[broken_df].index = pd.to_datetime(df_set[broken_df].index)

            behav_data[subj] = pd.concat(df_set).sort_index()
        else:
            print('data not found for %s' % (subj))
    if force_boolean:
        for subj in subjects:
            if subj in behav_data:
                for forced in force_boolean:
                    behav_data[subj][forced] = behav_data[subj][forced].map(lambda x: x in [True, 'True', 'true', 1, '1'])
    return behav_data

def _make_dt_maker(year):
    def dt_maker(x):
        return dt.datetime(year, int(x['old_date'][0:2]), int(x['old_date'][2:]), int(x['TimeOfDay'][0:2]), int(x['TimeOfDay'][2:]))
    return dt_maker

def _read_year_rDAT(rDat_f, nheaderrows):
    with open(rDat_f) as f:
        head = [next(f) for x in range(nheaderrows)]
    date_line = [x for x in head if 'Start time' in x]
    return int(date_line[0][-5:-1])
