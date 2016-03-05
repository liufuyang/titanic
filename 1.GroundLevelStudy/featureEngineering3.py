import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer


def feature_engineering(_df, get_dummies=False):
    title_mapping = {
        'Capt': 0.0,
        'Col': 0.5,
        'Don': 0.0,
        'Dr': 0.4285714285714286,
        'Jonkheer': 0.0,
        'Lady': 1.0,
        'Major': 0.5,
        'Master': 0.575,
        'Miss': 0.6978021978021978,
        'Mlle': 1.0,
        'Mme': 1.0,
        'Mr': 0.15667311411992269,
        'Mrs': 0.792,
        'Ms': 1.0,
        'Rev': 0.0,
        'Sir': 1.0,
        'the Countess': 1.0
    }
    cabin_mapping = {
        'A': 0.4666666666666667,
        'B': 0.7446808510638298,
        'C': 0.5932203389830508,
        'D': 0.7575757575757576,
        'E': 0.75,
        'F': 0.6153846153846154,
        'G': 0.5,
        'T': 0.0,
        'X': 0.29985443959243085
    }


    imputer_age = Imputer(missing_values='NaN', strategy='mean', axis=0)

    imputer_age.fit(_df[['Age']])
    _df['Sex_'] = _df['Sex'].apply(lambda x: 1 if x=='female' else 0)

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())

    _df['Fare_'] = _df['Fare'].fillna(20)
    _df['Fare_'] = _df['Fare'].apply(lambda x: 40 if x > 40 else x)
    _df['HasFare'] = _df['Fare'].apply(lambda x: 0 if np.isnan(x) else 1)

    _df['SibSp_'] = _df['SibSp'].apply(lambda x: 3 if x > 3 else x)
    _df['Parch_'] = _df['Parch'].apply(lambda x: 3 if x > 3 else x)
    _df['HasFamily'] = (_df['SibSp'] + _df['Parch']).map(lambda x: 0 if x == 0 else 1)

    # Age
    _df['HasAge'] = _df['Age'].apply(lambda x: 0 if np.isnan(x) else 1)
    _df['Age_'] = imputer_age.transform(_df['Age'].reshape(-1, 1))
    # or
    #_df['Age_'] = _df["Age"].fillna(_df["Age"].mean())
    # http://stackoverflow.com/questions/21050426/pandas-impute-nans

    _df['Age_b'] = np.digitize(_df['Age_'], [0,5,10,15,20,25,28,30,35,40,45,50,55,60,65,70])
    _df['IsChild'] = _df['Age'].map(lambda x: 1 if x < 16 else 0)

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    _df['Title_s'] = _df['Title_'].map(title_mapping)

    # Cabin:
    _df['Cabin_'] = _df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    _df['Cabin_s'] = _df['Cabin_'].map(cabin_mapping)
    # NaN is no problem for get_dummies
    # However let's try to keep it as a feature called X

    # Embarked:
    _df['Embarked_'] = _df['Embarked'].apply(lambda x: 'S' if isinstance(x, float) else x)

    #df_return = _df[['Age_','Age_b','HasAge','Sex','Pclass','Fare_','HasFare','Title_','Embarked_','Cabin_','SibSp_','Parch_']]
    df_return = _df[['Age_','HasAge', 'IsChild','Sex_','Pclass','Fare_','Title_s','Embarked_','Cabin_s', 'HasFamily']]


    if get_dummies:
        return pd.get_dummies(df_return)
    else:
        return df_return
