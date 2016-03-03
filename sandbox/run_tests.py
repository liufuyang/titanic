# This is the file you run to have the test results
# In console, run: python run_test.py

from test_model import test_model

from Model import Model
from Model import GenderModel

# def model_1(a, b):
#     return a + b
#
# result = test_model(model_1)
#
# print "Result of model_1 is %d." % result
model1 = Model()
result = test_model(model1, 50)

model2 = GenderModel()
result = test_model(model2, 50)
