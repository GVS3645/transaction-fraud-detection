import pandas as pd
import joblib
import os

class Fraud:
    def __init__(self):
        try:
            self.ohe = joblib.load("functions/onehotencoder_cycle1.joblib")
            self.scaler = joblib.load("functions/minmaxscaler_cycle1.joblib")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"🚨 Required file not found: {e}")
        except Exception as e:
            raise RuntimeError(f"🚨 Error loading encoder/scaler: {e}")

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df['errorbalanceOrg'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
        df['errorbalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
        return df

    def data_preparation(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_type = self.ohe.transform(df[['type']])
        except Exception as e:
            raise ValueError(f"🚨 OneHotEncoder failed: {e}")

        df_type_df = pd.DataFrame(df_type, columns=self.ohe.get_feature_names_out(['type']))
        df.reset_index(drop=True, inplace=True)
        df_type_df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df_type_df], axis=1)
        df.drop(columns=['type'], inplace=True)

        try:
            df_scaled = self.scaler.transform(df)
        except Exception as e:
            raise ValueError(f"🚨 Scaler failed: {e}")

        return pd.DataFrame(df_scaled, columns=df.columns)

    def get_prediction(self, model, original_data, test_data) -> pd.DataFrame:
        try:
            pred = model.predict(test_data)
        except Exception as e:
            raise RuntimeError(f"🚨 Prediction failed: {e}")

        original_data['prediction'] = pred
        return original_data
