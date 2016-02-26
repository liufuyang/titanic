
import numpy as np

def feature_engineering_train(df):

    df['Title'] = df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    df['Cabin_'] = df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    df['Embarked_'] = df['Embarked'].apply(lambda x: 'S' if isinstance(x, float) else x)

    title_mapping={}
    for t in np.unique(df['Title']):
        x = df[df['Title'] == t]
        title_mapping[t] = 1 - x['Survived'].sum()/float(len(x['Survived']))

    cabin_mapping={}
    for t in np.unique(df['Cabin_']):
        x = df[df['Cabin_'] == t]
        cabin_mapping[t] = 1 - x['Survived'].sum()/float(len(x['Survived']))

    embarked_mapping={}
    for t in np.unique(df['Embarked_']):
        x = df[df['Embarked_'] == t]
        embarked_mapping[t] = 1 - x['Survived'].sum()/float(1+len(x['Survived']))

    feature_engineering_test(df, title_mapping, cabin_mapping, embarked_mapping)

    return (title_mapping, cabin_mapping, embarked_mapping)

def feature_engineering_test(_df,title_mapping, cabin_mapping, embarked_mapping):

    _df['Sex_'] = _df['Sex'].apply(lambda x: 0.0 if x=='female' else 1)

    _df['Fare'] = _df['Fare'].fillna(10)
    _df['Fare'] = _df['Fare'].apply(lambda x: 40 if x > 40 else x)

    fareMax = _df['Fare'].max()
    fareMin = _df['Fare'].min()

    _df['Fare_s'] = (_df['Fare']-fareMin)/(fareMax-fareMin)

    _df['Pclass_s'] = _df['Pclass']/3

    _df['SibSp'] = _df['SibSp'].apply(lambda x: 3 if x > 3 else x)
    _df['SibSp_s'] = _df['SibSp']/_df['SibSp'].max()

    _df['Parch'] = _df['Parch'].apply(lambda x: 3 if x > 3 else x)
    _df['Parch_s'] = _df['Parch']/_df['Parch'].max()

    _df['HasAge'] = _df['Age'].apply(lambda x: 0 if np.isnan(x) else 1)

    _df['Age_'] = _df['Age'].fillna(29)
    _df['Age_b'] = np.digitize(_df['Age_'], [0,5,10,15,20,25,28,30,35,40,45,50,55,60,65,70])
    ageMax = _df['Age_b'].max()
    ageMin = _df['Age_b'].min()

    _df['Age_s'] = (_df['Age_b']-ageMin)/float((ageMax-ageMin))

    _df['Title'] = _df['Name'].apply(lambda x: x.replace('.',',').split(',')[1].strip())
    _df['Title_s'] = _df['Title'].map(title_mapping)
    _df['Title_s'] = _df['Title_s'].fillna(0);

    # Cabin:
    _df['Cabin_'] = _df['Cabin'].apply(lambda x: 'X' if isinstance(x, float) else x[0])
    _df['Cabin_s'] = _df['Cabin_'].map(cabin_mapping)

    # Embarked:
    _df['Embarked_'] = _df['Embarked'].apply(lambda x: 'S' if isinstance(x, float) else x)
    _df['Embarked_s'] = _df['Embarked_'].map(embarked_mapping)

    # Combined features
    _df['Sex+Age'] = _df['Age_s'] + _df['Sex_']
    _df['Pclass+Fare'] = _df['Pclass_s'] + _df['Fare_s']
