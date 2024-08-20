# from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# save file into folders
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb')as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)
    

def model_evaluate(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():

            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)

            # Store the result in the report dictionary
            report[model_name] = test_model_accuracy

        return report

    except Exception as e:
        raise CustomException(e, sys) 
    

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)