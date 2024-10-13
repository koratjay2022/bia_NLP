import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.base import is_classifier, is_regressor

class common_Fucnation : 
    
    # def __init__(self,data) :
    #     self.data = data
    def __init__(self) :
        pass
    
    def clean_dataframe(self, data):
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        return data

    def replace_values(self, data, column: str, to_replace: dict):
        data[column].replace(to_replace, inplace=True)
        return data

    def remove_rows(self, data, column: str, value):
        data = data[data[column] != value]
        return data

    def label_encode(self, data, column: str):
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        return data[column]

    def all_label_encode(self, data):
        le = LabelEncoder()
        object_counlm = data.select_dtypes(include=["object"]).columns
        for i in object_counlm :
             data[i] = le.fit_transform(data[i])
        return data

    def scale_features(self, data, columns: list):
        scaler = StandardScaler()
        data[columns] = scaler.fit_transform(data[columns])
        return data

    def remove_outliers(self, data, column: str, threshold: float = 1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        return data
        
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    if is_classifier(model):
        metric = accuracy_score(y_test, y_pred)
        print(f"Model Type: Classifier\nAccuracy: {metric}")
    elif is_regressor(model):
        metric = r2_score(y_test, y_pred)
        print(f"Model Type: Regressor\nR² Score: {metric}")
    else:
        raise ValueError("Model must be either a classifier or a regressor.")
    
    return y_pred,metric,model

def train_and_evaluate_multiple_models(models, X_train, X_test, y_train, y_test) :
    results = {}
    
    for model in models:
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        model_name = model.__class__.__name__
        
        if is_classifier(model):
            metric = accuracy_score(y_test, y_pred)
            print(f"Model: {model_name} (Classifier) -> Accuracy: {metric}")
        elif is_regressor(model):
            metric = r2_score(y_test, y_pred)
            print(f"Model: {model_name} (Regressor) -> R² Score: {metric}")
        else:
            raise ValueError(f"Model {model_name} is neither a classifier nor a regressor.")
        
        results[model_name] = metric
    
    return results
# class common_Fucnation : 
    
#     def __init__(self,data) :
#         self.data = data
    
    
#     def clean_dataframe(self):
#         """Drop null values and duplicates."""
#         self.data.dropna(inplace=True)
#         self.data.drop_duplicates(inplace=True)
#         return self.data

#     def replace_values(self, column: str, to_replace, value):
#         return self.data

#     def remove_rows(self, condition):
#         self.data = self.data.query(condition)
#         return self.data

#     def label_encode(self, column: str):
#         le = LabelEncoder()
#         self.data[column] = le.fit_transform(self.data[column])
#         return self.data

#     def all_label_encode(self):
#         le = LabelEncoder()
#         categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
#         for col in categorical_cols:
#             self.data[col] = le.fit_transform(self.data[col])
#         return self.data

#     def scale_features(self, columns: list):
#         scaler = StandardScaler()
#         self.data[columns] = scaler.fit_transform(self.data[columns])
#         return self.data

#     def remove_outliers(self, column: str, threshold: float = 1.5):
#         Q1 = self.data[column].quantile(0.25)
#         Q3 = self.data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - (threshold * IQR)
#         upper_bound = Q3 + (threshold * IQR)
#         self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
#         return self.data
        
#     def decluster(self, columns: list, n_neighbors: int = 20, contamination: float = 0.05):
#         lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
#         is_inlier = lof.fit_predict(self.data[columns]) > 0
#         self.data = self.data[is_inlier]
#         return self.data




# import pandas as pd
# from sklearn.preprocessing import StandardScaler,LabelEncoder
# import numpy as np


# class common_Fucnation :
#     def clean_dataframe(data, drop_nulls=False, drop_duplicates=False):
#         if drop_nulls:
#             data = data.dropna()
            
#         if drop_duplicates:
#             data = data.drop_duplicates()
            
#         return data
#     import pandas as pd

#     def replace_values(data, column, replace_dict):
#         data[column].replace(replace_dict, inplace=True)
#         return data

#     def remove_rows(data, column, remove_value):
#         data = data[data[column] != remove_value]
#         return data


#     def label_encode(data, column):
#         le = LabelEncoder()
#         data[column] = le.fit_transform(data[column])
#         return data

#     def scale_features(data, columns):
#         scaler = StandardScaler()
#         data[columns] = scaler.fit_transform(data[columns])
#         return data

#     def remove_outliers(data, column):
#         print(column)
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1

#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]



#     def handle_outliers(data, column, method='remove'):
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR

#         if method == 'remove':
#             return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
#         elif method == 'replace':
#             data[column] = np.where(
#                 (data[column] < lower_bound) | (data[column] > upper_bound),
#                 np.nan, data[column]
#             )
#             return data.fillna(data[column].median())
#         elif method == 'clip':
#             data[column] = np.clip(data[column], lower_bound, upper_bound)
#             return data
#         else:
#             raise ValueError("Method must be 'remove', 'replace', or 'clip'.")

