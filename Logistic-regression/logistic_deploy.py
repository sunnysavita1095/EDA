#Let's start with importing necessary libraries
import pickle
import bz2
import numpy as np
import pandas as pd

class predObj:

    def predict_log(self, dict_pred):
        
        scalarobject=bz2.BZ2File("Model\standardScalar.pkl", "rb")
        scaler=pickle.load(scalarobject)
        modelforpred = bz2.BZ2File("Model\modelForPrediction.pkl", "rb")
        model = pickle.load(modelforpred)

        data_df = pd.DataFrame(dict_pred,index=[1,])
        print(data_df)
        scaled_data = scaler.transform(data_df)
        predict = model.predict(scaled_data)
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'

        return result



