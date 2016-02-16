#Randomly Divide Training Data into train _data and _validation data (0.7, 0.3?)
#Train model with _train data and test it with _validation data, check the prediction correct rate (may do this multiple times and check the average accuracy)

# this function takes a model in and run test on it and return the
# accuracy of the model

import pandas as pd
import numpy as np

def test_model(model):

    df = pd.read_csv('data/train.csv', sep=',')

    msk = np.random.rand(len(df)) < 0.8
    _train = df[msk]
    _validation = df[~msk]

    print 'Training sample number: %d' % _train.shape[0]
    print 'Validation sample number: %d' % _validation.shape[0]

    model.learn(_train)

    predictions = _validation.apply(model.predict, axis=1);

    correct_ratio =  1 - (abs(predictions - _validation['Survived'])).sum()/float(len(_validation))

    print correct_ratio
    return correct_ratio
