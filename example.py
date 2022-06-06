#!/usr/bin/env python3
"""

Standard errors additional info:
Also see http://www.vincentgregoire.com/standard-errors-in-python/
"""

import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pytz
import statsmodels.api as sm
import statsmodels.formula.api as smf
import string
import sys

__projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

# function to create a tabular table
sys.path.append(str(__projectdir__ / Path('submodules/python-tabular-output/')))
from tab_general_func import tabularconvert
from tab_sm_func import getcoefftabmatrix
from tab_sm_func import getparamtabmatrix
from tab_sm_func import getsmresultstable
from tab_general_func import mergetabsecs

sys.path.append(str(__projectdir__ / Path('submodules/time-index-func/')))
from time_index_func import *

# Basic Data Manipulation:{{{1
def groupby_ex():
    df = pd.DataFrame({'group': ['a', 'a', 'b'], 'var1': [1, 2, 3], 'var2': [2, 3, 4]})
    # group every variable by mean
    print('groupby basic all variables')
    df2 = df.groupby(['group']).mean()
    print(df2)

    # group only var1 by mean
    print('groupby basic one variable')
    df2 = df.groupby(['group'])['var1'].mean()
    print(df2)

    # group by custom function
    print('groupby custom same length as original group')
    # where custom function defines same number of values as in original group
    def f(x):
        x['var1_demeanedbygroup'] = x['var1'] - x['var1'].mean()

        return(x)
    df2 = df.groupby(['group']).apply(f)
    print(df2)

    # group by custom function
    print('groupby custom same length one per group')
    # where custom function defines same number of values as in original group
    def f(x):
        return( np.mean(list(x['var1'])) - np.max(list(x['var1'])) )
    df2 = df.groupby(['group']).apply(f)
    print(df2)

    # or alternatively with lambda
    if False:
        df2 = df.groupby(['group']).apply(lambda x: np.mean(list(x['var1'])) - np.max(list(x['var1'])))
        print(df2)


def reshape_ex():

    # pivot basic:{{{
    # i is the index in the wide version
    # j is the name of the second index created from suffixes in the long version
    # stubnames are the variables that are being indexed i.e. the first part of the variables that are currently given across the second index
    df = pd.DataFrame({'gdp1': [100, 102], 'gdp2': [100, 101], 'year': [2000, 2001]})
    df2 = pd.wide_to_long(df, stubnames = 'gdp', i = 'year', j = 'country')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = 'gdp')
    df3.columns = ['gdp' + str(gdpcol) for gdpcol in df3.columns]
    df3 = df3.reset_index()
    print(df3)
    # pivot basic:}}}

    # pivot multiple variables:{{{
    # with multiple variables
    df = pd.DataFrame({'gdp1': [100, 102], 'gdp2': [100, 101], 'unemp1': [5, 4], 'unemp2': [4, 3], 'year': [2000, 2001]})
    df2 = pd.wide_to_long(df, stubnames = ['gdp', 'unemp'], i = 'year', j = 'country')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = ['gdp', 'unemp'])
    # remove multi-index
    df3.columns = [tup[0] + str(tup[1]) for tup in df3.columns]
    df3 = df3.reset_index()
    print(df3)
    # pivot multiple variables:}}}

    # pivot suffix not a number:{{{
    # need to specify if suffix is not a number using suffix = '\D+'
    # suffix using numbers (only) (default): suffix = '\d+'
    # suffix using letters (only): suffix = '\D+'
    # suffix using letters or numbers: suffix = '\S+'
    # adding separator
    # adding additional index variable
    df = pd.DataFrame({'gdp_usa': [100, 102], 'gdp_japan': [100, 101], 'unemp_usa': [5, 4], 'unemp_japan': [4, 3], 'year': [2000, 2001], 'worldgpd': [200, 203]})
    df2 = pd.wide_to_long(df, stubnames = ['gdp', 'unemp'], i = 'year', j = 'country', sep = '_', suffix = '\D+')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = ['gdp', 'unemp'])
    # with multiple values, get multi-index column that need to combine manually
    df3.columns = [tup[0] + '_' + tup[1] for tup in df3.columns]
    df3 = df3.reset_index()
    print(df3)
    # pivot suffix not a number:}}}

    # pivot_table multiple index:{{{
    # i is the index in the wide version
    # j is the name of the second index created from suffixes in the long version
    # stubnames are the variables that are being indexed i.e. the first part of the variables that are currently given across the second index
    df = pd.DataFrame({'source': ['AUS', 'CAN', 'AUS', 'CAN'], 'dest': ['AUS', 'AUS', 'CAN', 'CAN'], 'exports_2000': [1, 2, 3, 4], 'exports_2001': [2, 3, 4, 5]})
    df2 = pd.wide_to_long(df, stubnames = 'exports', i = ['source', 'dest'], j = 'year', sep = "_")
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot_table(df2, index = ['source', 'dest'], columns = 'year', values = 'exports')
    df3.columns = ['exports_' + str(gdpcol) for gdpcol in df3.columns]
    df3 = df3.reset_index()
    print(df3)
    # pivot_table multiple index:}}}

    # pivot_table multiple index/multiple values:{{{
    # i is the index in the wide version
    # j is the name of the second index created from suffixes in the long version
    # stubnames are the variables that are being indexed i.e. the first part of the variables that are currently given across the second index
    df = pd.DataFrame({'source': ['AUS', 'CAN', 'AUS', 'CAN'], 'dest': ['AUS', 'AUS', 'CAN', 'CAN'], 'exports_2000': [1, 2, 3, 4], 'exports_2001': [2, 3, 4, 5], 'imports_2000': [1, 2, 3, 4], 'imports_2001': [2, 3, 4, 5]})
    df2 = pd.wide_to_long(df, stubnames = ['imports', 'exports'], i = ['source', 'dest'], j = 'year', sep = "_")
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot_table(df2, index = ['source', 'dest'], columns = 'year', values = ['exports', 'imports'])
    # with multiple values, get multi-index column that need to combine manually
    df3.columns = [col[0] + '_' + str(col[1]) for col in df3.columns]
    df3 = df3.reset_index()
    print(df3)
    # pivot_table multiple index:}}}


# Dates:{{{1
def datetime_ex():

    # basics datetime:{{{
    # create date
    # 1/1/20
    print(datetime.date(2020, 1, 1))

    # create datetime
    # 1/1/20 at midnight
    print(datetime.datetime(2020, 1, 1))
    # 1/1/20 at midday
    print(datetime.datetime(2020, 1, 1, 12, 0))

    # get specific components of datetime:
    date1 = datetime.datetime(2020, 1, 2, 12, 30, 15)
    print(date1.year)
    print(date1.month)
    print(date1.day)
    print(date1.hour)
    print(date1.minute)
    print(date1.second)
    # returns the day of the week where Monday is 0 and Sunday is 6
    print(date1.weekday())

    # convert datetime to/from string
    date1 = datetime.datetime(2020, 1, 1, 12, 30, 15)
    # convert to string
    date2 = date1.strftime('%Y%m%d %H:%M:%S')
    print(date2)
    # convert to datetime
    date3 = datetime.datetime.strptime(date2, '%Y%m%d %H:%M:%S')
    print(date3)
    # basics datetime:}}}

    # relativedelta:{{{
    date1 = datetime.datetime(2020, 1, 1, 12, 0)

    # datetime.relativedelta can only take fixed time periods
    # so it can only take days/seconds/minutes/hours/weeks but not years/months
    date2 = date1 + datetime.timedelta(days = 2)
    print(date2)
    date2 = date1 + datetime.timedelta(minutes = 30)
    print(date2)

    # relativedelta from dateutil offers more functionality here
    date2 = date1 + relativedelta(months = 1)
    print(date2)
    date2 = date1 + relativedelta(years = 1)
    print(date2)
    # }}}

    # basics pandas timestamp:{{{
    # can get timestamp from string
    date1 = pd.Timestamp('2020-01-02 12:30:15')
    print(date1)

    # can get timestamp from separate inputs
    date1 = pd.Timestamp(2020, 1, 2, 12, 30, 15)
    print(date1)

    # attributes
    date1 = pd.Timestamp(2020, 1, 2, 12, 30, 15)
    print(date1.year)
    print(date1.month)
    print(date1.day)
    print(date1.hour)
    print(date1.minute)
    print(date1.second)
    # returns the day of the week where Monday is 0 and Sunday is 6
    print(date1.weekday())

    # convert to datetime
    date1 = pd.Timestamp('2020-01-02 12:30:15').to_pydatetime()
    print(date1)
    # convert back to pandas Timestamp
    date2 = pd.Timestamp(date1)
    print(date2)

    # convert to date
    date1 = pd.Timestamp('2020-01-02 12:30:15').date()
    print(date1)
    # convert back to pandas Timestamp
    date2 = pd.Timestamp(date1)
    print(date2)

    # basics pandas timestamp:}}}

    # list of dates/timestamps:{{{

    # get all dates from 1983-01-01 to 1983-01-07 including 1983-01-01 and 1983-01-07
    startdate = datetime.date(1983, 1, 1)
    enddate = datetime.date(1983, 1, 7)
    daydiff = (enddate - startdate).days
    dates = [datetime.date(startdate.year, startdate.month, startdate.day) + datetime.timedelta(days = day) for day in range(0, daydiff + 1)]
    print(dates)

    # same approach with datetime rather than date
    startdate = datetime.datetime(1983, 1, 1)
    enddate = datetime.datetime(1983, 1, 7)
    daydiff = (enddate - startdate).days
    dates = [datetime.datetime(startdate.year, startdate.month, startdate.day) + datetime.timedelta(days = day) for day in range(0, daydiff + 1)]
    print(dates)

    # get all dates from 1983-01-01 to 1983-01-07 including 1983-01-01 and 1983-01-07 with a frequency of every 2 days
    freq = 2
    startdate = datetime.date(1983, 1, 1)
    enddate = datetime.date(1983, 1, 7)
    daydiff = (enddate - startdate).days
    dates = [datetime.date(startdate.year, startdate.month, startdate.day) + datetime.timedelta(days = day) for day in range(0, daydiff + 1, freq)]
    print(dates)

    # pandas timestamp create
    # note uses american date format
    dates = pd.date_range(start = '1/1/1983', end = '1/3/1983')
    print(dates)
    dates = pd.date_range(start = '1/1/1983', periods = 3)
    print(dates)
    dates = pd.date_range(end = '1/3/1983', periods = 3)
    print(dates)
    dates = pd.date_range(start = '1/1/1983', end = '1/3/1983', freq = '1D')
    print(dates)

    # see all possible frequencies with date_range here: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

    # every 12 hours
    dates = pd.date_range(start = '1/1/1983', end = '1/3/1983', freq = '12H')
    print(dates)

    # every month
    dates = pd.date_range(start = '1/1/1983', end = '4/1/1983', freq = 'M')
    print(dates)

    # every quarter
    dates = pd.date_range(start = '1/1/1983', end = '10/1/1983', freq = 'Q')
    print(dates)

    # convert to datetime list from pandas timestamp list
    dates1 = pd.date_range(start = '1/1/1983', end = '1/3/1983')
    dates2 = dates1.to_pydatetime()
    print(dates2)
    # convert back to timestamp
    dates3 = [pd.Timestamp(date) for date in dates2]
    print(dates3)

    # list of dates/timestamps:}}}

    # convert timezone:{{{
    # mainly understood from:
    # https://howchoo.com/g/ywi5m2vkodk/working-with-datetime-objects-and-timezones-in-python

    # see list of all pytz timezones using pytz.all_timezones
    # and common timezones using pytz.common_timezones

    d = datetime.datetime.now()
    la = pytz.timezone("America/New_York")
    d_la = la.localize(d)
    print(d_la)
    print(d_la.tzinfo)

    d_utc = d_la.astimezone(pytz.timezone("GMT"))
    print(d_utc)
    d_la = d_utc.astimezone(la)
    print(d_la)

    # one line
    print(pytz.timezone("America/New_York").localize(datetime.datetime.now()).astimezone(pytz.timezone("GMT")))

    # convert timezone:}}}



def datetime_me_ex():
    # basic convert time
    mytime = '20100101d'
    dt = convertmytimetodatetime(mytime)
    print(dt)
    mytime = convertdatetimetomytime(dt, 'd')
    print(mytime)

    # add 10 days to a date
    periods = addperiodsbyfreq(datetime.datetime(2010, 1, 1), 'd', 10)
    print(periods)

    # points between datetime
    pointsbetween = getallpointsbetween(datetime.datetime(2010, 1, 1), datetime.datetime(2010, 1, 5), 'd')
    print(pointsbetween)

    # points between mytime
    pointsbetween = getallpointsbetween_mytime('20100101d', '20100105d')
    print(pointsbetween)

    # days of week
    dow = getdayofweek(getallpointsbetween_mytime('20100101d', '20100105d'))
    print(dow)

    # weekendsonly
    df = pd.DataFrame(index = getallpointsbetween_mytime('20100101d', '20100105d'))
    dfnoweekend = weekdaysonly(df)
    print(dfnoweekend)

    # fill time
    df = filltime(dfnoweekend)
    print(df)

    # probably don't use this
    # df = pd.DataFrame(index = ['20100101d', '20100201d', '20100202d'])
    # df = raisefreq(df, 'm')
    # print(df)


# Statsmodels:{{{1
def cross_section_ex():
    # get dataset:{{{

    # set random number seed
    np.random.seed(1)

    # Example of generating random number distribution
    # loc = mean, scale = sd, size = array of distribution
    x1 = np.random.normal(loc = 0, scale = 1, size = [100])
    # low = lower bound, high = upper bound, size = array of distribution
    x2 = np.random.uniform(low = 0, high = 1, size = [100])

    # dummies creation
    x3 = list(range(10))
    x3 = np.repeat(x3, 10)

    epsilon = np.random.normal(size = [100])

    y = 1 + x1 + 2 * x2 + epsilon

    dforiginal = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
    # get dataset:}}}

    # regression with matrices:{{{
    # Basic Regression where I create matrices
    df = dforiginal.copy()

    # to add dummies
    df2 = pd.get_dummies(df['x3'], prefix = 'x3dummy')
    df = pd.concat([df, df2], axis = 1)

    y = df['y']
    X = df[['x1', 'x2'] + [column for column in df.columns if column.startswith('x3dummy')]]

    X = sm.add_constant(X)

    model = sm.OLS(y, X, missing = 'drop').fit()
    print(model.summary())
    # regression with matrices:}}}

    # basic regressions with formulae:{{{
    df = dforiginal.copy()

    # basic
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit()

    # dummy variable
    model = smf.ols(formula = 'y ~ x1 + x2 + C(x3)', data=df).fit()

    # no constant
    model = smf.ols(formula = 'y ~ x1 + x2 -1', data=df).fit()

    # interaction
    # : includes only x1*x2
    model = smf.ols(formula = 'y ~ x1 : x2', data=df).fit()
    # * includes x1, x2, x1*x2
    model = smf.ols(formula = 'y ~ x1 * x2', data=df).fit()

    # apply logs
    model = smf.ols(formula = 'y ~ np.log(np.exp(x1)) + x2', data=df).fit()
    # basic regressions with formulae:}}}

    # different standard errors:{{{
    # Homoskedastic standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit()

    # Heteroskedastic (robust) standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit(cov_type = 'HC3')

    # Clustered standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit(cov_type = 'cluster', cov_kwds = {'groups': df['x3']})

    # HAC standard errors with Bartlett Kernel
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit(cov_type = 'HAC', cov_kwds = {'maxlags': 1})
    # different standard errors:}}}


def dummies_ex():
    # get dataset:{{{

    ni = 10
    nt = 1000
   
    # Example of generating random number distribution
    # loc = mean, scale = sd, size = array of distribution
    x1 = np.random.normal(loc = 0, scale = 1, size = [ni, nt])
    epsilon = np.random.normal(loc = 0, scale = 1, size = [ni, nt])

    # basic fixed effect
    alphai = np.random.normal(loc = 0, scale = 5, size = [ni])
    # time trend by entity
    gammai = np.random.normal(loc = 0, scale = 0.5, size = [ni])
    gammai = [0] * ni

    # time fixed effect
    gammat = np.random.normal(loc = 0, scale = 5, size = [nt])

    y = np.empty([ni, nt])
    idi = np.empty([ni, nt])
    timet = np.empty([ni, nt])
    for i in range(ni):
        for t in range(nt):
            y[i, t] = alphai[i] + gammat[t] + gammai[i] + x1[i, t] + epsilon[i, t]
            idi[i, t] = i
            timet[i, t] = t
    # reshape as variables
    y = y.reshape(ni * nt)
    x1 = x1.reshape(ni * nt)
    idi = idi.reshape(ni * nt)
    timet = timet.reshape(ni * nt)

    dforiginal = pd.DataFrame({'y': y, 'x1': x1, 'id': idi, 'time': timet})
    # get dataset:}}}

    df = dforiginal

    model0 = smf.ols(formula = 'y ~ x1', data = dforiginal).fit()

    model1 = smf.ols(formula = 'y ~ x1 + C(time)', data = dforiginal).fit()

    model2 = smf.ols(formula = 'y ~ x1 + C(time) + C(id)', data = dforiginal).fit()

    # multiplying and including other necessary terms for multiplication with dummy to make sense
    model3 = smf.ols(formula = 'y ~ x1 * C(id)', data = dforiginal).fit()
    print(model3.summary())
    
    # multiplying and including other necessary terms for multiplication with dummy to make sense
    model4 = smf.ols(formula = 'y ~ x1 : C(id)', data = dforiginal).fit()
    print(model4.summary())
    

# Me Tables:{{{1
def tabularconvert_me_ex():
    """
    Examples of my tables to convert listoflists into tabular
    """

    # basic:{{{
    tabular = tabularconvert([['Col1', 'Col2'], ['a', 'b'], ['1', '2']], colalign = '|l|r|', hlines = [0, 1, -1], savename = None)
    print(tabular)
    # basic:}}}

    # multicolumn:{{{
    tabular = tabularconvert([['\\multicolumn{2}{|c|}{Cols. 1 and 2}', 'Col3'], ['Col1', 'Col2', 'Col3'], ['a', 'b', 'c'], ['1', '2', '3']], colalign = '|c|c|c|', hlines = [0, 1, 2, -1], savename = None)
    print(tabular)
    # multicolumn:}}}

    # multirow:{{{
    tabular = tabularconvert([['', 'Col1', 'Col2'], ['\\multirow{2}{*}{Letters}', 'a', 'b'], ['', 'A', 'B'], ['Numbers', '1', '2']], colalign = 'lcc', hlines = [0, 1, 3, -1], savename = None)
    print(tabular)
    # multirow:}}}

    # regression example:{{{

    # data setup:{{{
    N = 1000
    Nfirsthalf = N // 2
    beta1 = 1
    beta2 = 1

    x1 = np.random.normal(size = N)
    x2 = np.random.normal(size = N)
    epsilon = np.random.normal(size = N)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'epsilon': epsilon})
    df['y1'] = beta1 * df['x1'] + beta2 * df['x2'] + df['epsilon']
    df['y2'] = 2 * beta1 * df['x1'] + 2 * beta2 * df['x2'] + 2 * df['epsilon']
    # data setup:}}}

    outputrows = []
    # add title row
    outputrows.append(['x/y', 'y1', 'y2'])

    # put x in rows and y in columns
    for x in ['x1', 'x2']:
        betarow = [x]
        stdnrow = ['']
        for y in ['y1', 'y2']:
            df2 = df[(df[y].notna()) & (df[x].notna())].copy()

            model = smf.ols(formula = y + ' ~ ' + x, data = df2).fit()

            beta = model.params[1]
            pval = model.pvalues[1]
            stderr = model.bse[1]
            # tstat = model.tvalues[1]
            n = len(df2)
            
            # add stars
            abeta = '{:.3f}'.format(beta)
            if pval < 0.05:
                abeta = abeta + '*'
            if pval < 0.01:
                abeta = abeta + '*'
            if pval < 0.001:
                abeta = abeta + '*'

            stderr = '{:.3f}'.format(stderr)
            n = str(n)

            betarow.append(abeta)
            stdnrow.append('(' + stderr + ',' + n + ')')
        
        outputrows.append(betarow)
        outputrows.append(stdnrow)

    tabular = tabularconvert(outputrows, colalign = 'lcc', hlines = [0, 1, -1], savename = None)
    print(tabular)
    # regression example:}}}


def getsmresultstable_me_ex():
    """
    Examples of my functions to convert statsmodels into regression tables.
    """
    # data setup:{{{
    N = 1000
    Nfirsthalf = N // 2
    beta1 = 1
    beta2 = 1

    x1 = np.random.normal(size = N)
    x2 = np.random.normal(size = N)
    epsilon = np.random.normal(size = N)

    df = pd.DataFrame({'x1': x1, 'x2': x2, 'epsilon': epsilon})
    df['y'] = beta1 * df['x1'] + beta2 * df['x2'] + df['epsilon']

    df['firsthalf'] = 0
    df.loc[df.index[: Nfirsthalf], 'firsthalf'] = 1

    dffirsthalf = df[df['firsthalf'] == 1].copy()
    dfsecondhalf = df[df['firsthalf'] == 0].copy()
    # data setup:}}}

    # basic reg output:{{{
    model0 = smf.ols(formula = 'y ~ x1', data = df).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = df).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = df).fit()
    models = [model0, model1, model2]
    tabular = getsmresultstable(models, printtab = True, savename = None, ynames = None, coefflist = None, coeffnames = None)
    # basic reg output:}}}
    
    # adjust what x shown:{{{
    model0 = smf.ols(formula = 'y ~ x1', data = df).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = df).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = df).fit()
    models = [model0, model1, model2]

    # only show these x variables
    coefflist = ['x1', 'x2']
    # rename the x variables by the dict
    coeffnames = {'x1': 'x1 var'}

    tabular = getsmresultstable(models, printtab = True, savename = None, ynames = None, coefflist = coefflist, coeffnames = coeffnames)
    # adjusting what x shown:}}}
    
    # adjust what y shown:{{{
    model0 = smf.ols(formula = 'y ~ x1', data = df).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = df).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = df).fit()
    models = [model0, model1, model2]

    tabular = getsmresultstable(models, printtab = True, savename = None, ynames = ['Yname', 'y', 'y', 'y'])
    # adjusting what y shown:}}}
    
    # adding afterlofl:{{{
    model0 = smf.ols(formula = 'y ~ x1', data = df).fit()
    model1 = smf.ols(formula = 'y ~ x1', data = dffirsthalf).fit()
    model2 = smf.ols(formula = 'y ~ x1', data = dfsecondhalf).fit()
    models = [model0, model1, model2]

    tabular = getsmresultstable(models, printtab = True, savename = None, afterlofl = [['Data', 'All', 'First Half', 'Second Half']])
    # adding afterlofl:}}}
    
    # params adjust:{{{
    model0 = smf.ols(formula = 'y ~ x1', data = df).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = df).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = df).fit()
    models = [model0, model1, model2]
    tabular = getsmresultstable(models, printtab = True, savename = None, paramlist = ['nobs', 'rsquared'], paramnames = ['N', '$R^2$'], paramdecimal = [0, 3])
    # params adjust:}}}

    # multiple panels:{{{
    numreg = 3
    ynamesmatrix = [['', '(1)', '(2)', '(3)']]

    paneltabs = []
    panelnames = []

    # first half
    model0 = smf.ols(formula = 'y ~ x1', data = dffirsthalf).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = dffirsthalf).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = dffirsthalf).fit()
    models = [model0, model1, model2]
    paneltabs.append( getcoefftabmatrix(models) + getparamtabmatrix(models) )
    panelnames.append('First Half')

    # second half
    model0 = smf.ols(formula = 'y ~ x1', data = dfsecondhalf).fit()
    model1 = smf.ols(formula = 'y ~ x2', data = dfsecondhalf).fit()
    model2 = smf.ols(formula = 'y ~ x1 + x2', data = dfsecondhalf).fit()
    models = [model0, model1, model2]
    paneltabs.append( getcoefftabmatrix(models) + getparamtabmatrix(models) )
    panelnames.append('Second Half')

    tabsecs = []
    tabsecs.append(tabularconvert(ynamesmatrix))

    for i in range(len(paneltabs)):
        paneltitle = [[''] + ['\\multicolumn{' + str(numreg) + '}{c}{Panel ' + string.ascii_lowercase[i].upper() + ': ' + panelnames[i] + '}']]
        tabsecs.append( tabularconvert(paneltitle + paneltabs[i]) )

    tex = mergetabsecs(tabsecs, colalign = 'l' + 'c' * numreg, hlines = 'all', savename = None)
    print(tex)
    # multiple panels:}}}
    
# Matplotlib:{{{1
def lines_ex():
    xval = list(range(-3, 3))
    y0val = [x ** 2 for x in xval]
    y1val = [x ** 2 + 1 for x in xval]
    y2val = [x ** 2 + 2 for x in xval]

    # black dashed
    plt.plot(xval, y0val, 'k--')
    # thick green
    plt.plot(xval, y1val, 'g', linewidth = 4)
    # yellow dotted
    plt.plot(xval, y2val, 'y:')
    plt.show()
    plt.clf()


def legendposition_ex():

    # legend different position graph:{{{
    plt.plot([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], label = 'hello')
    plt.plot([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], label = 'goodbye')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower right')
    plt.show()
    plt.clf()
    # legend different position graph:}}}

    # legend below graph:{{{
    plt.plot([1, 2, 3, 4, 5], [2, 3, 4, 5, 6], label = 'hello')
    plt.plot([1, 2, 3, 4, 5], [3, 4, 5, 6, 7], label = 'goodbye')
    # ncol determines number of columns in legend
    # adjust y in bbox_to_anchor to move up and down
    # fontsize argument not needed
    fontsize = 10
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(prop = {'size': fontsize}, loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox = True, shadow = True, ncol = 2)
    # need to adjust after bbox_to_anchor
    plt.tight_layout()
    plt.show()
    plt.clf()
    # legend below graph:}}}


def scatterplot_ex():
    
    # get variables
    np.random.seed(1)
    x = np.random.normal(size = [100])
    u = np.random.normal(size = [100])
    desc = list(range(100))
    y = x + u

    df = pd.DataFrame({'x': x, 'y': y, 'desc': desc})
    
    # basic plot
    xvar = 'x'
    yvar = 'y'
    plt.scatter(df[xvar], df[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    # plt.savefig(LOCATION)
    plt.show()
    plt.clf()

    # scatter with line of best fit
    xvar = 'x'
    yvar = 'y'

    plt.scatter(df[xvar], df[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)

    # line of best fit
    x = df[xvar].values
    y = df[yvar].values
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m * x + b, 'k', linewidth = 4)

    # plt.savefig(LOCATION)
    plt.show()
    plt.clf()

    # plot with annotated column
    xvar = 'x'
    yvar = 'y'
    plt.scatter(df[xvar], df[yvar])
    plt.xlabel(xvar)
    plt.ylabel(yvar)
    # add annotation for each month
    for i, txt in enumerate(df['desc']):
        plt.annotate(txt, (df[xvar][i], df[yvar][i]), size = 5)
    plt.show()
    plt.clf()


# Full:{{{1
def full():
    groupby_ex()
    datetime_ex()
    reshape_ex()

    cross_section_ex()
    dummies_ex()

    lines_ex()
    legendposition_ex()
    scatterplot_ex()


# Run:{{{1
if __name__ == "__main__":
    full()
