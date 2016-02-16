# This is the file you run to have the test results
# In console, run: python run_test.py

from test_model import test_model

def model_1(a, b):
    return a + b

result = test_model(model_1)

print "Result of model_1 is %d." % result
