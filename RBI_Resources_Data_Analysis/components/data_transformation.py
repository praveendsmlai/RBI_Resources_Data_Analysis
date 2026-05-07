import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer

from RBI_Resources_Data_Analysis.constant.training_pipeline import TARGET_COLUMN,CAT_COLUMNS,NUM_COLUMNS
from RBI_Resources_Data_Analysis.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from RBI_Resources_Data_Analysis.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from RBI_Resources_Data_Analysis.entity.config_entity import DataTransformationConfig
from RBI_Resources_Data_Analysis.exception.exception import CustomException 
from RBI_Resources_Data_Analysis.logging.logger import logging
from RBI_Resources_Data_Analysis.utils.main_utils.utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise CustomException(e,sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
  
    def feature_engineering(self,df):
        df = df.copy()


        df = df.drop(index=0,axis=0)
        df=df.drop(columns=["Unnamed: 0"],axis=1)
        df = df.drop(columns=["Fortnight Ended"],axis=0)

        df["All Investments"] = df["Investments in Commercial Paper"]+df["Investments in shares"]+df["Investments in Bonds/Debentures"]+df["Total Non-SLR Investments"]

        df=df.drop(columns=["Investments in Commercial Paper","Investments in shares","Investments in Bonds/Debentures","Total Non-SLR Investments"])

        df.columns = df.columns.str.strip()

        return df
        
    def get_data_transformer_object(self)->Pipeline:
        """
        Creates preprocessing pipeline for numerical and categorical features
        """

        logging.info("Entered get_data_transformer_object method")

        try:
             # -------------------------
                # Numerical Pipeline
             # -------------------------
            
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
                #("scaler",StandardScaler())
            ])

                # -------------------------
                    # Categorical Pipeline
                # -------------------------
            # cat_pipeline = Pipeline([
            #     ("imputer", SimpleImputer(strategy="most_frequent")),
            #     ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
            # ])

                # -------------------------
                    # Column Transformer
                # -------------------------
            preprocessor = ColumnTransformer([
                ("num", num_pipeline, NUM_COLUMNS)
                #("cat", cat_pipeline, CAT_COLUMNS)
            ])

            feature_transformer = FunctionTransformer(self.feature_engineering,validate=False)

            final_pipeline = Pipeline([
                (
                    "feature_engineering",
                    feature_transformer
                ),

                (
                    "preprocessing",
                    preprocessor
                )
            ])

            logging.info("Preprocessing pipeline created successfully")

            return final_pipeline
        except Exception as e:
            raise CustomException(e, sys)  


        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = np.log1p(target_feature_train_df)

            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = np.log1p(target_feature_test_df)

            preprocessor=self.get_data_transformer_object()

            #preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor.transform(input_feature_test_df)

            print(type(transformed_input_train_feature))
            print(transformed_input_train_feature.shape)

            print(type(target_feature_train_df))
            print(target_feature_train_df.shape)
             

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            # train_arr = np.hstack((
            #                 transformed_input_train_feature.toarray(),
            #                     np.array(target_feature_train_df).reshape(-1, 1)
            # ))

            # test_arr = np.hstack((
            #                 transformed_input_test_feature.toarray(),
            #                 np.array(target_feature_test_df).reshape(-1, 1)
            # ))

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor)


            os.makedirs("final_model", exist_ok=True)
            save_object( "final_model/preprocessor.pkl", preprocessor,)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise CustomException(e,sys)
