#Randomly Divide Training Data into train _data and _validation data (0.7, 0.3?)
#Train model with _train data and test it with _validation data, check the prediction correct rate (may do this multiple times and check the average accuracy)

# this function takes a module_function in and run test on it and return the
# accuracy of the model

def test_model(model):
    print('Test object is {:s}'.format(type(model)))
    print('Test result is {:d}'.format(model(6,7)))
    return model(6, 7)
