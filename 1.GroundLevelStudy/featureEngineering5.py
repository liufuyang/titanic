import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA


def feature_engineering_step1(_df):
    title_mapping = {
        'Capt': 0.0,
        'Col': 0.2,
        'Don': 0.0,
        'Dr': 0.2,
        'Jonkheer': 0.0,
        'Lady': 1.0,
        'Major': 0.2,
        'Master': 0.8,
        'Miss': 0.9,
        'Mlle': 1.0,
        'Mme': 1.0,
        'Mr': 0.1,
        'Mrs': 0.8,
        'Ms': 1.0,
        'Rev': 0.0,
        'Sir': 0.0,
        'the Countess': 0.0
    }
    title_age_mapping = {
        'Capt': 70,
        'Col': 58,
        'Don': 40,
        'Dr': 40,
        'Jonkheer': 40,
        'Lady': 50,
        'Major': 50,
        'Master': 1,
        'Miss': 5,
        'Mlle': 5,
        'Mme': 24,
        'Mr': 30,
        'Mrs': 35,
        'Ms': 28,
        'Rev': 43,
        'Sir': 49,
        'the Countess': 30
    }
    cabin_mapping = {
        'A': 0.5,
        'B': 0.8,
        'C': 0.6,
        'D': 0.8,
        'E': 0.8,
        'F': 0.6,
        'G': 0.5,
        'T': 0.0,
        'X': 0.3
    }


    imputer_age = Imputer(missing_values='NaN', strategy='mean', axis=0)

    imputer_age.fit(_df[['Age']])
    _df['Sex_'] = _df['Sex'].apply(lambda x: 1 if x=='female' else 0)

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())

    #_df['Fare_'] = _df['Fare'].fillna(20)

    #_df['Fare_'] = _df['Fare_'].apply(lambda x: 40 if x > 40 else x)

    ####
    _df['Fare_'] = _df['Fare']
    _df.loc[ (_df.Fare.isnull())&(_df.Pclass==1),'Fare_'] =np.median(_df[_df['Pclass'] == 1]['Fare'].dropna())
    _df.loc[ (_df.Fare.isnull())&(_df.Pclass==2),'Fare_'] =np.median( _df[_df['Pclass'] == 2]['Fare'].dropna())
    _df.loc[ (_df.Fare.isnull())&(_df.Pclass==3),'Fare_'] = np.median(_df[_df['Pclass'] == 3]['Fare'].dropna())
    ####
    _df['Fare_'] = _df['Fare_'] / (1+_df['SibSp']+_df['Parch'])
    _df['HasFare'] = _df['Fare'].apply(lambda x: 0 if np.isnan(x) else 1)


    _df['SibSp_'] = _df['SibSp'].apply(lambda x: 3 if x > 3 else x)
    _df['Parch_'] = _df['Parch'].apply(lambda x: 3 if x > 3 else x)
    _df['FamilySize'] = (_df['SibSp'] + _df['Parch'])
    _df['HasFamily'] = (_df['SibSp'] + _df['Parch']).map(lambda x: 0 if x == 0 else 1)

    # Age
    _df['HasAge'] = _df['Age'].apply(lambda x: 0 if np.isnan(x) else 1)
    _df['Age_'] = imputer_age.transform(_df['Age'].reshape(-1, 1))

    # or
    #_df['Age_'] = _df["Age"].fillna(_df["Age"].mean())
    # http://stackoverflow.com/questions/21050426/pandas-impute-nans

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    _df.loc[(_df['Title_'].isnull()) & (_df['Sex']=='female'),('Title_')] = 'Miss'
    _df.loc[(_df['Title_'].isnull()) & (_df['Sex']=='male' ), ('Title_')] = 'Master'
    #_df['Title_s'] = _df['Title_s'].fillna(1.0)

    _df['Title_s'] = _df['Title_'].map(title_mapping)

    _df['Title_Age_s'] = _df['Title_'].map(title_age_mapping)
    _df['Title_Age_s'] = _df['Title_Age_s'].fillna(30)

    # Cabin:
    _df['Cabin_'] = _df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    _df['Cabin_s'] = _df['Cabin_'].map(cabin_mapping)
    # NaN is no problem for get_dummies
    # However let's try to keep it as a feature called X

    # Embarked:
    _df['Embarked_'] = _df['Embarked'].apply(lambda x: 'S' if isinstance(x, float) else x)



    #df_return = _df[['Age_','Age_b','HasAge','Sex','Pclass','Fare_','HasFare','Title_','Embarked_','Cabin_','SibSp_','Parch_']]
    df_return = _df[['Age','Age_','HasAge', 'Sex_', 'Sex','Pclass','Fare_', 'Title_s', 'Title_',
                     'Title_Age_s','Embarked_','Cabin_s', 'HasFamily', 'SibSp_','Parch_','FamilySize']]

    return pd.get_dummies(df_return)
#############################################


def feature_engineering(df_train, df_test):
    df_d_train = feature_engineering_step1(df_train)
    df_d_test = feature_engineering_step1(df_test)

    df_d_train_HasAge = df_d_train[df_d_train['HasAge']==1]
    df_d_test_HasAge = df_d_test[df_d_test['HasAge']==1]

    df_d_HasAge = pd.concat([df_d_train_HasAge, df_d_test_HasAge])

    #df_d_HasAge = df_d_train_HasAge

    features_age=['Sex_', 'Sex_female','Sex_male', 'Title_Age_s', 'Cabin_s', 'Embarked__C','Embarked__Q','Embarked__S','SibSp_','Parch_','Fare_','Pclass']

    X_train = df_d_HasAge[features_age]
    y_train = df_d_HasAge['Age_']

    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    pca = PCA(n_components=50)
    poly = PolynomialFeatures(degree=6)
    lr = LinearRegression(n_jobs=-1)

    scaler.fit(X_train)

    X_train_std = scaler.transform(X_train)
    X_train_poly = poly.fit_transform(X_train_std)
    X_train_poly = pca.fit_transform(X_train_poly)

    lr.fit(X_train_poly, y_train)

    # Predict for all

    X_predict_train_poly = poly.transform(scaler.transform(df_d_train[features_age]))
    X_predict_train_poly = pca.transform(X_predict_train_poly)
    df_d_train['Age_P'] = (lr.predict(X_predict_train_poly) + df_d_train['Title_Age_s'])/2
    df_d_train['Age_P'] = df_d_train['Age_P'].apply(lambda x: 0 if x<0 else x).apply(lambda x: 80 if x>80 else x)


    X_predict_test_poly = poly.transform(scaler.transform(df_d_test[features_age]))
    X_predict_test_poly = pca.transform(X_predict_test_poly)
    df_d_test['Age_P'] = (lr.predict(X_predict_test_poly) + df_d_test['Title_Age_s'])/2
    df_d_test['Age_P']=df_d_test['Age_P'].apply(lambda x: 0 if x<0 else x).apply(lambda x: 80 if x>80 else x)


    # Fill in Age_ as Age_P

    #df_d_train.loc[df_d_train['HasAge']==0, ('Age_')]= df_d_train[df_d_train['HasAge']==0]['Age_P']
    #df_d_test.loc[df_d_test['HasAge']==0, ('Age_')]= df_d_test[df_d_test['HasAge']==0]['Age_P']
    df_d_train.loc[df_d_train['HasAge']==0, ('Age_')]= df_d_train[df_d_train['HasAge']==0]['Title_Age_s']
    df_d_test.loc[df_d_test['HasAge']==0, ('Age_')]= df_d_test[df_d_test['HasAge']==0]['Title_Age_s']



    df_d_train['IsChild'] = df_d_train['Age_'].map(lambda x: 1 if x < 16 else 0)
    df_d_test['IsChild'] = df_d_test['Age_'].map(lambda x: 1 if x < 16 else 0)

    df_d_train['Fare_b'] = np.digitize(df_d_train['Fare_'], [0,5,10,20,30,40])
    df_d_test['Fare_b'] = np.digitize(df_d_test['Fare_'], [0,5,10,20,30,40])

    df_d_train['Age_b'] = np.digitize(df_d_train['Age_'], [0,15,25,40,60,80])
    df_d_test['Age_b'] = np.digitize(df_d_test['Age_'], [0,15,25,40,60,80])

    df_d_train['AgeCat']=df_d_train['Age_']
    df_d_train.loc[ (df_d_train.Age_<=14) ,'AgeCat'] = 'child'
    df_d_train.loc[ (df_d_train.Age_>60),'AgeCat'] = 'aged'
    df_d_train.loc[ (df_d_train.Age_>14) & (df_d_train.Age_ <=30) ,'AgeCat'] = 'adult'
    df_d_train.loc[ (df_d_train.Age_>30) & (df_d_train.Age_ <=60) ,'AgeCat'] = 'senior'

    df_d_test['AgeCat']=df_d_test['Age_']
    df_d_test.loc[ (df_d_test.Age_<=14) ,'AgeCat'] = 'child'
    df_d_test.loc[ (df_d_test.Age_>60),'AgeCat'] = 'aged'
    df_d_test.loc[ (df_d_test.Age_>14) & (df_d_test.Age_ <=30) ,'AgeCat'] = 'adult'
    df_d_test.loc[ (df_d_test.Age_>30) & (df_d_test.Age_ <=60) ,'AgeCat'] = 'senior'

    del df_d_train['Age_P']
    del df_d_test['Age_P']

    return pd.get_dummies(df_d_train), pd.get_dummies(df_d_test)
