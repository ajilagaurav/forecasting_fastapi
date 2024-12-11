import pandas as pd
import numpy as np

class ForecastRepository:
    
    def __init__(self, dataset):
        self.dataset = dataset

    def analyze_columns(self):
        sample_df = self.dataset.sample(min(1000, len(self.dataset)))  # Sample up to 1000 rows
        
        # Detect date columns
        date_columns = sample_df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
        detected_date_columns = []
        for col in date_columns:
            try:
                sample_df[col] = pd.to_datetime(sample_df[col], errors='coerce')
                if sample_df[col].notna().sum() > 0:
                    detected_date_columns.append(col)
            except:
                continue

        # Detect numeric columns
        numeric_columns = sample_df.select_dtypes(include=[np.number]).columns.tolist()

        # Detect categorical columns (non-numeric, non-date columns with few unique values)
        categorical_columns = sample_df.select_dtypes(include=['object']).nunique()[lambda x: x < 50].index.tolist()
        
        return {
            "date_columns": detected_date_columns,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns
        }
