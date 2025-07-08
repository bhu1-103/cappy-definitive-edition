
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

def load_data():
    """Loads and merges input and output data."""
    input_files = sorted(glob.glob('/home/bhu1/dev/git/cappy-definitive-edition/step2/z_output/*.csv'))
    output_files = sorted(glob.glob('/home/bhu1/dev/git/cappy-definitive-edition/step4/sce1a_output/throughput/*.csv'))

    if not input_files or not output_files:
        raise FileNotFoundError("Input or output files are missing.")

    input_dfs = []
    for i in range(len(input_files)):
        try:
            input_df = pd.read_csv(input_files[i], sep=';')
            output_df = pd.read_csv(output_files[i], header=None)
            
            # The output CSV is a single row of comma-separated values.
            # The number of values should correspond to the number of rows in the input CSV.
            if len(input_df) == output_df.shape[1]:
                input_df['throughput'] = output_df.T.iloc[:, 0]
                input_dfs.append(input_df)
            else:
                print(f"Skipping file pair due to mismatched lengths: {input_files[i]} and {output_files[i]}")

        except Exception as e:
            print(f"Error processing file pair: {input_files[i]}, {output_files[i]}. Error: {e}")


    if not input_dfs:
        raise ValueError("No matching input and output files could be processed. Please check file alignment and format.")

    combined_df = pd.concat(input_dfs, ignore_index=True)
    return combined_df

def find_significant_features(df):
    """Preprocesses the data and finds the most significant features."""
    if 'throughput' not in df.columns:
        raise ValueError("Target column 'throughput' not found in the dataframe.")

    X = df.drop('throughput', axis=1)
    y = df['throughput']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns

    # Create a preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with feature selection
    # We'll select all numerical features to see their scores
    fs_pipeline = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('feature_selection', SelectKBest(f_regression, k='all'))
    ])

    # Execute the pipeline
    fs_pipeline.fit(X, y)

    # Get the scores
    scores = fs_pipeline.named_steps['feature_selection'].scores_

    # Get feature names after one-hot encoding
    try:
        cat_feature_names = fs_pipeline.named_steps['preprocessing'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    except AttributeError: # Older sklearn versions
        cat_feature_names = fs_pipeline.named_steps['preprocessing'].named_transformers_['cat'].get_feature_names(categorical_features)

    all_feature_names = list(numerical_features) + list(cat_feature_names)

    # Create a dataframe of features and their scores
    feature_scores = pd.DataFrame({'Feature': all_feature_names, 'Score': scores})
    feature_scores = feature_scores.sort_values(by='Score', ascending=False)

    print("Feature Importance Scores (F-scores):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(feature_scores)

if __name__ == '__main__':
    try:
        dataframe = load_data()
        find_significant_features(dataframe)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")

