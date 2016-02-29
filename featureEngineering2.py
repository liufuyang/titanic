import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

def feature_engineering(_df, get_dummies=False):

    imputer_age = Imputer(missing_values='NaN', strategy='mean', axis=0)

    imputer_age.fit(_df[['Age']])

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())

    _df['Fare_'] = _df['Fare'].fillna(10)
    _df['Fare_'] = _df['Fare'].apply(lambda x: 40 if x > 40 else x)
    _df['HasFare'] = _df['Fare_'].apply(lambda x: 0 if np.isnan(x) else 1)

    _df['SibSp_'] = _df['SibSp'].apply(lambda x: 3 if x > 3 else x)
    _df['Parch_'] = _df['Parch'].apply(lambda x: 3 if x > 3 else x)

    # Age
    _df['HasAge'] = _df['Age'].apply(lambda x: 0 if np.isnan(x) else 1)
    _df['Age_'] = imputer_age.transform(_df['Age'].reshape(-1, 1))
    # or
    #_df['Age_'] = _df["Age"].fillna(_df["Age"].mean())
    # http://stackoverflow.com/questions/21050426/pandas-impute-nans

    _df['Age_b'] = np.digitize(_df['Age_'], [0,5,10,15,20,25,28,30,35,40,45,50,55,60,65,70])

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())

    # Cabin:
    _df['Cabin_'] = _df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    # NaN is no problem for get_dummies
    # However let's try to keep it as a feature called X

    # Embarked:
    _df['Embarked_'] = _df['Embarked'].apply(lambda x: 'X' if isinstance(x, float) else x)

    df_return = _df[['Age_','Age_b','HasAge','Sex','Pclass','Fare_','HasFare','Title_','Embarked_','Cabin_','SibSp_','Parch_']]


    if get_dummies:
        return pd.get_dummies(df_return)
    else:
        return df_return
