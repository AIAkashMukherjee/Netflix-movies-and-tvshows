import sys
import pandas as pd
import numpy as np
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
# from sklearn.feature_extraction.text import TfidfVectorizer
from src.entity.config import DataTransformationConfig
from src.utlis.utlis import save_obj

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation(self):
        try:
            str_columns=[ 'title', 'director', 'cast', 'country', 'release_year',
       'rating', 'duration', 'listed_in', 'description']
            
    #         ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
    #    'rating', 'duration', 'listed_in', 'description']
            
            cat_pipeline=Pipeline(
                steps = [
                # ("imputer", SimpleImputer(strategy = 'median')),
                # ("imputer", SimpleImputer(strategy='most_frequent')),
                # ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                # ("label_encoder", LabelEncoder()),
                ('scaler',StandardScaler())
                ]
            )
            preprocessor=ColumnTransformer([
                ('num_pipeline', cat_pipeline,str_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
        
    def transform_genres(self, df):
        # Convert 'genres' list column into a format suitable for encoding
        mlb = MultiLabelBinarizer()
        genres_transformed = mlb.fit_transform(df['genres'])
        genres_df = pd.DataFrame(genres_transformed, columns=mlb.classes_, index=df.index)
        return pd.concat([df.drop(columns=['genres']), genres_df], axis=1)
    
        
    def clean_data(self,df):
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(['show_id','date_added'],inplace=True,axis=1)
        return df
        
    def initate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            # Apply the clean_data method to both train and test datasets
            train_data = self.clean_data(train_data)
            test_data = self.clean_data(test_data)

            target_columns = 'type'
            preprocessor_obj = self.get_data_transformation()

            train_data['genres'] = train_data['listed_in'].str.split(',')
            test_data['genres'] = test_data['listed_in'].str.split(',')
            train_data = self.transform_genres(train_data)
            test_data = self.transform_genres(test_data)

            logger.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(columns=[target_columns], axis=1)
            target_feature_train_data = train_data[target_columns]

            logger.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(columns=[target_columns], axis=1)
            target_feature_test_data = test_data[target_columns]

            print("Train Data Columns:", input_feature_train_data.columns)
            print("Test Data Columns:", input_feature_test_data.columns)

            str_columns=[ 'title', 'director', 'cast', 'country', 'release_year',
            'rating', 'duration', 'listed_in', 'description', 'genres']
            
            # le=LabelEncoder()
            # for col in str_columns:
            #     input_train_arr=le.fit_transform(train_data[col].astype('str'))
            #     input_test_arr=le.fit_transform(test_data[col].astype('str'))

    
            print(input_feature_train_data.head(2))

            logger.info('Applying preprocessing')

            # Apply preprocessor object on our train data and test data
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

            logger.info('Applying Label encoder')

            target_encoder=LabelEncoder()
            target_encoder.fit(target_feature_train_data)
            target_feature_train_data = target_encoder.transform(target_feature_train_data)
            target_feature_test_data = target_encoder.transform(target_feature_test_data)

            logger.info('Combining into array')

            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessor_obj)

            logger.info('Exited from data transformation')

            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)