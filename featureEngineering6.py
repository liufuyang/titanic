import numpy as np
import pandas as pd

def feature_engineering_step1(_df):
    title_mapping = {
        'Capt': 'Mr',
        'Col': 'Mr',
        'Don': 'Mr',
        'Dr': 'Mr',
        'Jonkheer': 'Mr',
        'Lady': 'Mrs',
        'Major': 'Mr',
        'Master': 'Master',
        'Miss': 'Miss',
        'Mlle': 'Miss',
        'Mme': 'Mrs',
        'Mr': 'Mr',
        'Mrs': 'Mrs',
        'Ms': 'Miss',
        'Rev': 'Mr',
        'Sir': 'Mr',
        'the Countess': 'Mrs'
    }
    title_age_mapping = {
        'Capt': 'elder',
        'Col': 'elder',
        'Don': 'adult',
        'Dr': 'adult',
        'Jonkheer': 'adult',
        'Lady': 'elder',
        'Major': 'elder',
        'Master': 'young',
        'Miss': 'young',
        'Mlle': 'young',
        'Mme': 'adult',
        'Mr': 'adult',
        'Mrs': 'adult',
        'Ms': 'adult',
        'Rev': 'adult',
        'Sir': 'elder',
        'the Countess': 'adult'
    }
    cabin_mapping = {
        'A': 'M',
        'B': 'G',
        'C': 'M',
        'D': 'G',
        'E': 'G',
        'F': 'G',
        'G': 'M',
        'T': 'X',
        'X': 'X'
    }

    _df['Sex_'] = _df['Sex'].apply(lambda x: 1 if x=='female' else 0)

    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    _df['FamilyName'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[0].strip())

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

    _df['Fare_b'] = np.digitize(_df['Fare_'], [0,5,10,20,30,40])

    # Family Size
    _df['FamilySize'] = (_df['SibSp'] + _df['Parch'])
    _df['HasFamily'] = (_df['SibSp'] + _df['Parch']).map(lambda x: 0 if x == 0 else 1)

    # Age
    _df['HasAge'] = _df['Age'].apply(lambda x: 0 if np.isnan(x) else 1)
    _df['Age_s'] = _df['Age'].apply(age_to_s)

    # or
    #_df['Age_'] = _df["Age"].fillna(_df["Age"].mean())
    # http://stackoverflow.com/questions/21050426/pandas-impute-nans

    # Title
    _df['Title_'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    _df.loc[(_df['Title_'].isnull()) & (_df['Sex']=='female'),('Title_')] = 'Miss'
    _df.loc[(_df['Title_'].isnull()) & (_df['Sex']=='male' ), ('Title_')] = 'Master'

    _df['Title_s'] = _df['Title_'].map(title_mapping)

    _df['Title_Age_s'] = _df['Title_'].map(title_age_mapping)
    _df['Title_Age_s'] = _df['Title_Age_s'].fillna('adult')

    ## fill age NAN:
    _df.loc[_df['HasAge']==0, ('Age_s')]= _df[_df['HasAge']==0]['Title_Age_s']

    # Cabin:
    _df['Cabin_'] = _df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    _df['Cabin_s'] = _df['Cabin_'].map(cabin_mapping)
    # NaN is no problem for get_dummies
    # However let's try to keep it as a feature called X

    # Embarked:
    _df['Embarked_'] = _df['Embarked'].apply(lambda x: 'S' if isinstance(x, float) else x)


    df_return = _df.loc[:,('Age','Age_s','HasAge', 'Sex','Pclass','Fare_', 'Fare_b','Title_s',
                     'Title_Age_s','Embarked_','Cabin_s', 'HasFamily', 'SibSp','Parch','FamilySize','FamilyName')]

    return df_return
#############################################
def age_to_s(x):
    if x<=16:
        return 'young'
    elif x>16 and x<=40:
        return'adult'
    else:
        return'elder'

def feature_engineering(df_train, df_test):
    df_d_train = feature_engineering_step1(df_train)
    df_d_test = feature_engineering_step1(df_test)

    df_d_train_survivedFamily = df_d_train[ (df_train['Survived']==1) & (df_d_train['FamilySize']>0)]
    df_d_train_notSurvivedFamily = df_d_train[ (df_train['Survived']==0) & (df_d_train['FamilySize']>0)]
    #print df_d_train_survivedFamily
    survivedFamilyNames = df_d_train_survivedFamily['FamilyName']
    notSurvivedFamilyNames = df_d_train_notSurvivedFamily['FamilyName']

    # df_d_train.loc[:,('FamilySurvived')] = df_d_train[df_d_train['FamilySize']>0]['FamilyName'].apply(lambda x: 1 if x in survivedFamilyNames.values else 0)
    # df_d_test.loc[:,('FamilySurvived')] = df_d_test[df_d_test['FamilySize']>0]['FamilyName'].apply(lambda x: 1 if x in survivedFamilyNames.values else 0)
    #
    # df_d_train.loc[:,('FamilyDied')] = df_d_train[df_d_train['FamilySize']>0]['FamilyName'].apply(lambda x: 1 if x in notSurvivedFamilyNames.values else 0)
    # df_d_test.loc[:,('FamilyDied')] = df_d_test[df_d_test['FamilySize']>0]['FamilyName'].apply(lambda x: 1 if x in notSurvivedFamilyNames.values else 0)

    df_d_train.loc[:,('FamilySurvived')] = df_d_train[df_d_train['FamilySize']>0]['FamilyName'].apply(lambda x: (x == survivedFamilyNames.values).sum())
    df_d_test.loc[:,('FamilySurvived')] = df_d_test[df_d_test['FamilySize']>0]['FamilyName'].apply(lambda x: (x == survivedFamilyNames.values).sum())

    df_d_train.loc[:,('FamilyDied')] = df_d_train[df_d_train['FamilySize']>0]['FamilyName'].apply(lambda x: (x == notSurvivedFamilyNames.values).sum())
    df_d_test.loc[:,('FamilyDied')] = df_d_test[df_d_test['FamilySize']>0]['FamilyName'].apply(lambda x: (x == notSurvivedFamilyNames.values).sum())

    df_d_train['FamilySurvived'] = df_d_train['FamilySurvived'].fillna(0);
    df_d_train['FamilyDied']=df_d_train['FamilyDied'].fillna(0);
    df_d_test['FamilySurvived']=df_d_test['FamilySurvived'].fillna(0);
    df_d_test['FamilyDied']=df_d_test['FamilyDied'].fillna(0);

    del df_d_train['FamilyName']
    del df_d_test['FamilyName']

    return pd.get_dummies(df_d_train), pd.get_dummies(df_d_test)
