import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    df = pd.read_table("https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data")

    X = df.iloc[:,0:10]
    y = df.iloc[:,10]
    [length, column_number] = np.shape(X)

    print('正規化しない場合')
    fit(X, y)
    print('正規化した場合')
    fit(norm(X, X.describe().transpose()), y)

def fit(x, y):
    X2 = sm.add_constant(x)
    model = sm.OLS(y, X2)
    result = model.fit()
    print(result.summary())

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

if __name__ == '__main__':
    main()