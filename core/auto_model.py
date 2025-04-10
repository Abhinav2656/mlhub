# core/auto_model.py
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score

# Candidate models will be chosen based on task type
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Visualization functions from our existing file
from .visualizations import plot_confusion_matrix, plot_roc_curve

def auto_train_and_predict(df):
    """
    Automatically detects the target column, preprocesses data,
    trains candidate models, selects the best model using CV,
    fits on entire data, generates predictions, and produces visualizations.
    
    Assumes the target is the last column in the CSV.
    Returns a context dict with preview table HTML, visualizations filenames,
    target column, task type, best model name, and CV score.
    """
    # Assume target is the last column
    if df.empty or df.shape[1] < 2:
        raise ValueError("The CSV must have at least one feature column and one target column.")
    
    target_column = df.columns[-1]
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Remove columns that have a single unique value (e.g., IDs)
    constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)
    
    # Identify numeric and categorical columns in X
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    # Build preprocessor
    preprocessor = ColumnTransformer(transformers=[
         ('num', Pipeline(steps=[
             ('imputer', SimpleImputer(strategy='mean')),
             ('scaler', StandardScaler())
         ]), numeric_features),
         ('cat', Pipeline(steps=[
             ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
             ('onehot', OneHotEncoder(handle_unknown='ignore'))
         ]), categorical_features)
    ])
    
    # Infer task type
    # If target is object, or numeric with very few unique values (< 10), consider classification.
    if y.dtype == 'object' or y.nunique() < 10:
        task_type = 'classification'
    else:
        task_type = 'regression'
    # Even if numeric, treat as classification if very few unique values
    if y.dtype in ['int64','float64'] and y.nunique() < 10:
        task_type = 'classification'
    
    # Prepare candidate models
    if task_type == 'regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Random Forest Regressor': RandomForestRegressor(random_state=42)
        }
        scoring = 'neg_mean_squared_error'
    else:
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest Classifier': RandomForestClassifier(random_state=42),
            'K-Nearest Neighbors': KNeighborsClassifier()
        }
        scoring = 'accuracy'
    
    # Evaluate candidate models using cross-validation
    best_score = None
    best_model_name = None
    best_pipeline = None
    
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        try:
            scores = cross_val_score(pipe, X, y, cv=5, scoring=scoring)
            avg_score = scores.mean()
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue
        
        if best_score is None or avg_score > best_score:
            best_score = avg_score
            best_model_name = name
            best_pipeline = pipe
    
    # Check that a valid model was selected.
    if best_pipeline is None:
        raise ValueError("No candidate model was successfully trained. Please check your CSV file and ensure it has suitable features and target data.")
    
    # Fit the chosen pipeline on the entire data and generate predictions
    best_pipeline.fit(X, y)
    predictions = best_pipeline.predict(X)
    
    # Prepare a preview table: append predictions to original data
    result_df = X.copy()
    result_df[target_column] = y
    result_df['prediction'] = predictions
    table_html = result_df.head(10).to_html(classes='table table-striped')
    
    # Prepare visualizations (store files in "media" folder)
    vis_files = {}
    os.makedirs('media', exist_ok=True)
    
    if task_type == 'classification':
        # Generate confusion matrix and ROC if possible
        y_pred = predictions
        try:
            y_proba = best_pipeline.predict_proba(X)[:, 1]
        except Exception:
            y_proba = None
        
        try:
            vis_files['conf_matrix_img'] = plot_confusion_matrix(y, y_pred)
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
        
        if y_proba is not None:
            try:
                vis_files['roc_curve_img'] = plot_roc_curve(y, y_proba)
            except Exception as e:
                print(f"Error generating ROC curve: {e}")
    else:
        # For regression, generate a scatter plot of True vs Predicted
        try:
            plt.figure()
            plt.scatter(y, predictions, alpha=0.5)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title('True vs Predicted Values')
            filename = f"scatter_{uuid.uuid4().hex}.png"
            path = os.path.join('media', filename)
            plt.savefig(path)
            plt.close()
            vis_files['scatter_img'] = filename
        except Exception as e:
            print(f"Error generating regression plot: {e}")
    
    # Return context dictionary with details
    return {
         'target_column': target_column,
         'task_type': task_type,
         'best_model': best_model_name,
         'cv_score': best_score,
         'table': table_html,
         **vis_files
    }
