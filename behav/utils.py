import datetime as dt

def stars(p):
    '''Converts p-values into R-styled stars.

    Signif. codes:
        '***' :  < 0.001
        '**' : < 0.01
        '*' : < 0.05
        '.' : < 0.1
        'n.s.' : < 1.0

    '''
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '.'
    else:
        return 'n.s.'

def binP(N, p, x1, x2):
    p = float(p)
    q = p/(1-p)
    k = 0.0
    v = 1.0
    s = 0.0
    tot = 0.0

    while(k<=N):
            tot += v
            if(k >= x1 and k <= x2):
                    s += v
            if(tot > 10**30):
                    s = s/10**30
                    tot = tot/10**30
                    v = v/10**30
            k += 1
            v = v*q*(N+1-k)/k
    return s/tot

def binomial_ci(x,N,CL=95.0):
    '''
    Calculate the exact confidence interval for a binomial proportion

    from http://stackoverflow.com/questions/13059011/is-there-any-python-function-library-for-calculate-binomial-confidence-intervals

    Parameters:
    -----------
    x : int
        count of items
    N : int
        total number of items
    CL : float
        confidence limit

    Returns:
    --------
    tuple of floats
        the lower and upper bounds on the confidence interval

    Usage:
    >>> calcBin(13,100)
    (0.07107391357421874, 0.21204372406005856)
    >>> calcBin(4,7)
    (0.18405151367187494, 0.9010086059570312)
    '''
    x = float(x)
    N = float(N)
    #Set the confidence bounds
    TU = (100 - float(CL))/2
    TL = TU

    P = x/N
    if (x==0):
        dl = 0.0
    else:
        v = P/2
        sL = 0
        sH = P
        p = TL/100

        while((sH-sL) > 10**-5):
            if(binP(N, v, x, N) > p):
                sH = v
                v = (sL+v)/2
            else:
                sL = v
                v = (v+sH)/2
        dl = v

    if (x==N):
        ul = 1.0
    else:
        v = (1+P)/2
        sL = P
        sH = 1
        p = TU/100
        while((sH-sL) > 10**-5):
            if(binP(N, v, 0, x) < p):
                sH = v
                v = (sL+v)/2
            else:
                sL = v
                v = (v+sH)/2
        ul = v
    return (dl, ul)

def filter_normal_trials(df):
    '''
    filters dataframe, df, to only include normal (non-correction) trials that got a response.
    '''
    return df[(df.response!='none')&(df.type_=='normal')]

def filter_recent_days(df, num_days):
    '''
    filters dataframe, df, to only include the most recent num_days of trials
    '''
    today = dt.datetime.now()
    return df[(today.date()-dt.timedelta(days=num_days)):today]

def extract_filename(data_to_analyze, target='stim_name', inplace=True):
    if not inplace:
        data_to_analyze = data_to_analyze.copy()
    split_names = data_to_analyze.stimulus.str.split('/', expand=True)
    data_to_analyze[target] = split_names[list(split_names.keys()).values.max()].str.split('.', expand=True)[0]
    return data_to_analyze
