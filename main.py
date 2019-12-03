import numpy as np
import pandas as pd
import statsmodels.api as sm

def main():
    df = pd.read_table("https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data")

    X = df.iloc[:,0:10]
    y = df.iloc[:,10]
    normed_X = norm(X, X.describe().transpose())
    #normed_X = X
    [length, column_number] = np.shape(normed_X)
    X2 = sm.add_constant(normed_X)
    
    model = sm.OLS(y, X2)
    result = model.fit()
    print(result.summary())

def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']

if __name__ == '__main__':
    main()