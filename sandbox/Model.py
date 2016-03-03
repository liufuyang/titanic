# -*- coding: utf-8 -*-
# from some_preprocessing_module import some_prerocessing_tools

class Model:
    'Abstract Model Class'

    def learn(self, data):
        self.data = data

    def predict(self, data_row):
        return 0; # predict all people died

class GenderModel(Model):
    'Gender Model'

    def predict(self, data_row):

        if data_row['Sex'] == 'female':
            return 1 # predict all women survived
        else:
            return 0
