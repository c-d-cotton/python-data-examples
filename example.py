#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from pathlib import Path
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys

# __projectdir__ = Path(os.path.dirname(os.path.realpath(__file__)) + '/')

def cross_section(printsummary = False):
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

    # Regressions from Formulae
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

    # Homoskedastic standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit()

    # Heteroskedastic (robust) standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit(cov_type = 'HC3')

    # Clustered standard errors
    model = smf.ols(formula = 'y ~ x1 + x2', data=df).fit(cov_type = 'cluster', cov_kwds = {'groups': df['x3']})

    # group every variable by mean
    df2 = df.groupby(['x3']).mean()

    # group y by mean
    df2 = df.groupby(['x3'])['y'].mean()

    # group by custom function
    def f(x):
        x['x1_demeanedbyx3'] = x['x1'] - x['x1'].mean()

        return(x)
    df2 = df.groupby(['x3']).apply(f)

cross_section(printsummary = False)
