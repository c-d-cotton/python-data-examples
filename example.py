#!/usr/bin/env python3
"""
Also see http://www.vincentgregoire.com/standard-errors-in-python/
"""

import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

# __projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

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


cross_section(printsummary = False)
def groupby():
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
