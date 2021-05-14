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
import sys

# __projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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


# groupby()
def datetime_ex(doall = False):

    # basics datetime:{{{
    if False or doall is True:
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
    if False or doall is True:
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
    if False or doall is True:
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
    if False or doall is True:

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
    if True or doall is True:
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

# datetime_ex(doall = False)
def reshape():
    # i is the index in the wide version
    # j is the name of the second index created from suffixes in the long version
    # stubnames are the variables that are being indexed i.e. the first part of the variables that are currently given across the second index
    df = pd.DataFrame({'gdp1': [100, 102], 'gdp2': [100, 101], 'year': [2000, 2001]})
    df2 = pd.wide_to_long(df, 'gdp', i = 'year', j = 'country')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = 'gdp')
    df3.columns = ['gdp' + str(gdpcol) for gdpcol in df3.columns]
    df3 = df3.reset_index()
    print(df3)

    # with multiple variables
    df = pd.DataFrame({'gdp1': [100, 102], 'gdp2': [100, 101], 'unemp1': [5, 4], 'unemp2': [4, 3], 'year': [2000, 2001]})
    df2 = pd.wide_to_long(df, ['gdp', 'unemp'], i = 'year', j = 'country')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = ['gdp', 'unemp'])
    # remove multi-index
    df3.columns = [tup[0] + str(tup[1]) for tup in df3.columns]
    df3 = df3.reset_index()
    print(df3)

    # need to specify if suffix is not a number using suffix = '\D+'
    # adding separator
    # adding additional index variable
    df = pd.DataFrame({'gdp_usa': [100, 102], 'gdp_japan': [100, 101], 'unemp_usa': [5, 4], 'unemp_japan': [4, 3], 'year': [2000, 2001], 'worldgpd': [200, 203]})
    df2 = pd.wide_to_long(df, ['gdp', 'unemp'], i = 'year', j = 'country', sep = '_', suffix = '\D+')
    df2 = df2.reset_index()
    print(df2)
    # unpivot
    df3 = pd.pivot(df2, index = 'year', columns = 'country', values = ['gdp', 'unemp'])
    # remove multi-index
    df3.columns = [tup[0] + '_' + tup[1] for tup in df3.columns]
    df3 = df3.reset_index()
    print(df3)


# reshape()
# Statsmodels:{{{1
def cross_section(printsummary = False):
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
    if printsummary is True:
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

# cross_section(printsummary = False)
def dummies(doall = False):
    # get dataset:{{{

    ni = 1000
   
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
    for i in range(len(ni)):
        for t in range(len(nt)):
            y[i, t] = alphai[i] + gammat[t] + gammai[i] * t + x1[i, t] + epsilon[i, t]
            idi[i, t] = i
            timet[i, t] = t
    # reshape as variables
    y = y.reshape(ni * nt)
    idi = idi.reshape(ni * nt)
    timet = timet.reshape(ni * nt)

    dforiginal = pd.DataFrame({'y': y, 'x1': x1, 'id': idi, 'time': timet})
    # get dataset:}}}

    df = dforiginal

    model0 = smf.ols(formula = 'y ~ x1').fit()

    model1 = smf.ols(formula = 'y ~ x1 + C(time)').fit()

    model2 = smf.ols(formula = 'y ~ x1 + C(time) + C(id)').fit()

    





# Matplotlib:{{{1
def lines_ex(doall = False):
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

# lines_ex()
def scatterplot_ex(doall = False):
    
    # get variables
    np.random.seed(1)
    x = np.random.normal(size = [100])
    u = np.random.normal(size = [100])
    desc = list(range(100))
    y = x + u

    df = pd.DataFrame({'x': x, 'y': y, 'desc': desc})
    
    # basic plot
    if False or doall is True:
        xvar = 'x'
        yvar = 'y'
        plt.scatter(df[xvar], df[yvar])
        plt.xlabel(xvar)
        plt.ylabel(yvar)
        # plt.savefig(LOCATION)
        plt.show()
        plt.clf()

    # scatter with line of best fit
    if True or doall is True:
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
    if False or doall is True:
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

# scatterplot_ex(doall = False)

