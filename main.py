import numpy as np
import pandas as pd
import statsmodels.api as sm
import itertools

def main():
    df = pd.read_table("https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data")

    X = df.drop('Y', 1)
    y = df['Y']
    column_names = get_columns_list(X)
    combinations = get_combinations(column_names)

    print('正規化しない場合')
    fit(X, y)
    print('正規化した場合')
    result = fit(norm(X, X.describe().transpose()), y)
    print(result.summary())

    aic_min = 100000
    aic_min_index = 0
    bic_min = 10000
    bic_min_index = 0

    for i, combination in enumerate(combinations):
        X_new = X.loc[:, combination]
        result = fit(norm(X_new, X_new.describe().transpose()), y)
        if aic_min > result.aic:
            aic_min = result.aic
            aic_min_index = i
        if bic_min > result.bic:
            bic_min = result.bic
            bic_min_index = i
        print(f'{i}回目')

    print(f'AIC: {combinations[aic_min_index]}')
    print(f'BIC: {combinations[bic_min_index]}')
        
def fit(x, y):
    X2 = sm.add_constant(x)
    model = sm.OLS(y, X2)
    result = model.fit()
    return result

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

def get_columns_list(x):
    result = []
    columns = x.columns
    for i in range(columns.size):
        result.append(columns[i])
    return result

def get_combinations(columns):
    result = []
    for i in range(1, len(columns) + 1):
        for c in itertools.combinations(columns, i):
            result.append(list(c))
    return result

if __name__ == '__main__':
    main()