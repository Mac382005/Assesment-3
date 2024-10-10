import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Import necessary libraries for machine learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

import joblib


# Caching expensive operations
@st.cache_resource
def train_models(X_train_scaled, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "SVM Regressor": SVR()
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
    return trained_models


@st.cache_resource
def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


# Function to clean dataset by converting object columns to appropriate data types
def clean_dataset(data):
    # Step 1: Convert object columns to numeric where possible, but keep original value if conversion fails
    for col in data.select_dtypes(include=['object']).columns:
        # Try to convert to numeric, if not possible keep the original value
        data[col] = data[col].apply(lambda x: pd.to_numeric(x, errors='ignore') if isinstance(x, str) else x)

    # Step 2: Convert columns that may be dates, but keep original value if conversion fails
    for col in data.select_dtypes(include=['object']).columns:
        # Try to convert to datetime, if not possible keep the original value
        try:
            data[col] = pd.to_datetime(data[col], errors='ignore')
        except:
            pass  # Ignore errors during datetime conversion

    # Step 3: Infer remaining object columns
    data = data.infer_objects()

    # Step 4: Return the cleaned dataset
    return data

# Sidebar for file uploads
st.sidebar.title("Upload Dataset(s)")
uploaded_files = st.sidebar.file_uploader("Choose one or more CSV files", type="csv", accept_multiple_files=True)

# Dictionary to hold multiple datasets
datasets = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read each uploaded dataset and store it in a dictionary with the filename as the key
        datasets[uploaded_file.name] = pd.read_csv(uploaded_file)

    # Select dataset from uploaded files
    selected_file = st.sidebar.selectbox("Select a dataset to proceed with:", options=list(datasets.keys()))

    # Select the dataset to proceed with
    data = datasets[selected_file]
    data_cleaned = data.drop_duplicates()

    # Clean the dataset (convert object columns to numeric or datetime)
    # data_cleaned = clean_dataset(data_cleaned)


    # Main section for displaying and processing the selected dataset
    st.title(f"Processing Dataset: {selected_file}")

    # Step 1: Display a sample of the selected dataset
    st.write("### Sample Data from the Selected Dataset")
    st.dataframe(data_cleaned.sample(5), use_container_width=True)
    # Check and drop "Unnamed: 0" column if it exists
    if "Unnamed: 0" in data_cleaned.columns:
        data_cleaned = data_cleaned.drop(columns=["Unnamed: 0"])
        st.write("Dropped 'Unnamed: 0' column from the dataset.")
        st.dataframe(data_cleaned.sample(5), use_container_width=True)

    # Display the cleaned data types
    st.write("### Cleaned Data Types")
    dtype_df = pd.DataFrame(data_cleaned.dtypes, columns=["Data Type"]).reset_index().rename(
        columns={"index": "Column Name"})
    st.dataframe(dtype_df, use_container_width=True)

    # Display the shape and columns of the selected dataset
    st.write("### Dataset Information")
    st.write(f"### Shape: `{data_cleaned.shape}`")
    st.write("Columns in the dataset:", data_cleaned.columns.tolist())

    # Step 2: Problem Statement Definition
    target = st.selectbox("Select the target variable (dependent variable):", data_cleaned.columns)
    st.write(f"### Selected Target Variable: `{target}`")

    st.write(data_cleaned.Color.value_counts())
    data_cleaned.Size = data_cleaned.Size.map({'XS': 0, 'S': 1, 'M': 2, 'L': 3, 'XL': 4, 'XXL': 5})
    data_cleaned.Color = data_cleaned.Color.map({'Yellow': 20, 'White': 21, 'Red': 22, 'Black': 23, 'Blue': 24, 'Green': 25})
    data_cleaned.Category = data_cleaned.Category.map({'Jeans': 10, 'Shoes': 11, 'Dress': 12, 'Jacket': 13, 'Sweater': 14, 'T-shirt': 15})
    data_cleaned.Brand = data_cleaned.Brand.map({'Adidas': 30, 'Nike': 31, 'Puma': 32, 'Reebok': 33, 'Under Armour': 34, 'New Balance': 35})
    data_cleaned.Material = data_cleaned.Material.map({'Cotton': 40, 'Silk': 41, 'Polyester': 42, 'Denim': 43, 'Nylon': 44, 'Wool': 45})
    st.dataframe(data_cleaned.head(10))
    st.dataframe(data_cleaned.dtypes)
    # Step 3 - Visualizing the Target Variable
    st.write("## Step 3: Visualizing the Target Variable")

    fig, ax = plt.subplots()
    ax.hist(data_cleaned[target], bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Plot a pairplot of all numeric columns in the dataset
    numeric_cols = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 1:
        sns.pairplot(data_cleaned[numeric_cols].dropna())
        plt.title(f'Pairplot of Numeric Variables')
        st.pyplot(plt.gcf())
    else:
        st.write("Not enough numeric variables to plot a pairplot.")

    # Step 4: Basic Data Exploration
    st.write("## Step 4: Data Exploration")

    # Create two columns for data types and summary statistics
    col1, col2 = st.columns(2)

    # Column 1: Data Types
    with col1:
        st.write("### Data Types:")
        # Create a DataFrame for the column names and their corresponding data types
        dtype_df = pd.DataFrame(data_cleaned.dtypes, columns=["Data Type"]).reset_index()
        dtype_df = dtype_df.rename(columns={"index": "Column Name"})

        st.dataframe(dtype_df, use_container_width=True)

    # Column 2: Summary Statistics
    with col2:
        st.write("### Summary Statistics:")
        st.dataframe(data_cleaned.describe(), use_container_width=True)

    # Step 5: Visual EDA - Histograms for Continuous Variables
    st.write("## Step 5: Visual EDA - Histograms of Continuous Variables")
    continuous_columns = st.multiselect("Select continuous variables to visualize:",
                                        data_cleaned.select_dtypes(include=['float64', 'int64']).columns)

    if continuous_columns:
        for column in continuous_columns:
            # Create a new figure for each column
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.hist(data_cleaned[column], bins=30, edgecolor='k', alpha=0.7)
            ax.set_title(f"Distribution of {column}")
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

            # Display the individual figure
            st.pyplot(fig)

    # Step 6: Outlier Analysis
    st.write("## Step 6: Outlier Analysis")

    # Select only continuous numerical columns
    numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])

    if not numeric_columns.empty:
        # Proceed with the outlier analysis if there are numeric columns
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1

        # Identifying outliers using the IQR method
        outliers = (numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))

        # Count outliers in each column
        outlier_count = outliers.sum()

        # Display the number of outliers in each column
        st.write("### Number of Outliers in Each Numeric Column")
        dtype_df_outlier = pd.DataFrame(outlier_count, columns=["Number of Outliers"]).reset_index()
        dtype_df_outlier = dtype_df_outlier.rename(columns={"index": "Column Name"})
        st.dataframe(dtype_df_outlier, use_container_width=True)
    else:
        st.write("No continuous numeric columns available for outlier analysis.")

    # Step 7: Missing Value Analysis
    st.write("## Step 7: Missing Value Analysis")
    missing_values = data_cleaned.isnull().sum()
    st.write("### Missing Values in Each Column")
    dtype_df_missing_values = pd.DataFrame(missing_values, columns=["Missing Values"]).reset_index()
    dtype_df_missing_values = dtype_df_missing_values.rename(columns={"index": "Column Name"})
    st.dataframe(dtype_df_missing_values, use_container_width=True)

    # Step 8: Feature Selection - Correlation Analysis
    st.write("## Step 8: Feature Selection - Correlation Matrix")

    # Select only continuous numerical columns for correlation analysis
    numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])

    if not numeric_columns.empty:
        # Calculate and display the correlation matrix
        correlation_matrix = numeric_columns.corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        # Display the full correlation matrix
        st.write("### Correlation Matrix:")
        st.dataframe(correlation_matrix, use_container_width=True)

        # Extract correlations with the target variable
        target_correlations = correlation_matrix[target].drop(target)  # Drop correlation of target with itself

        # Display the correlation of each feature with the target variable
        st.write(f"### Correlation of Features with Target Variable: `{target}`")
        st.dataframe(target_correlations, use_container_width=True)

        # Identify strong, moderate, and weak relationships
        strong_corr = target_correlations[target_correlations.abs() >= 0.05]
        moderate_corr = target_correlations[(target_correlations.abs() >= 0.02) & (target_correlations.abs() < 0.05)]
        weak_corr = target_correlations[target_correlations.abs() < 0.02]

        # Display the correlations classified as strong, moderate, and weak
        st.write("### Strong Correlations (|correlation| ≥ 0.05):")
        st.dataframe(strong_corr, use_container_width=True)

        st.write("### Moderate Correlations (0.02 ≤ |correlation| < 0.05):")
        st.dataframe(moderate_corr, use_container_width=True)

        st.write("### Weak Correlations (|correlation| < 0.02):")
        st.dataframe(weak_corr, use_container_width=True)
    else:
        st.write("No continuous numeric columns available for correlation analysis.")

    # Step 9: Statistical Feature Selection using ANOVA for Categorical Variables
    st.write("## Step 9: Statistical Feature Selection (ANOVA for Categorical Variables)")

    # Select categorical columns for ANOVA analysis
    categorical_columns = data_cleaned.select_dtypes(include=['object'])

    if not categorical_columns.empty:
        selected_categorical = st.multiselect("Select categorical variables for ANOVA:", categorical_columns.columns)

        # Ensure the target variable is continuous
        if pd.api.types.is_numeric_dtype(data_cleaned[target]):
            # Dictionary to store ANOVA results
            anova_results = []

            # Perform ANOVA for each selected categorical variable
            for cat_col in selected_categorical:
                anova_groups = data_cleaned.groupby(cat_col)[target].apply(list)
                f_val, p_val = stats.f_oneway(*anova_groups)

                # Append the results to a list
                anova_results.append({"Categorical Variable": cat_col, "F-value": f_val, "p-value": p_val})

            # Convert results to DataFrame
            anova_df = pd.DataFrame(anova_results)

            # Display the ANOVA results
            st.write("### ANOVA Results")
            st.write("The following table shows F-values and P-values for each categorical variable.")
            st.dataframe(anova_df, use_container_width=True)

            # Display significant variables based on p-value
            if "p-value" in anova_df.columns:
                significant_vars = anova_df[anova_df["p-value"] < 0.05]
                st.write("### Significant Variables (p < 0.05):")
                if not significant_vars.empty:
                    st.dataframe(significant_vars, use_container_width=True)
                else:
                    st.write("No significant variables found.")
                # Visualize the relationship using box plots
                st.write("### Box Plot: Categorical Variable vs Target")
                for cat_col in selected_categorical:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=cat_col, y=target, data=data_cleaned, ax=ax)
                    ax.set_title(f"{cat_col} vs {target}")
                    st.pyplot(fig)
            else:
                st.write("No categorical variables selected for ANOVA.")

        else:
            st.write("The target variable must be continuous for ANOVA analysis.")
    else:
        st.write("No categorical variables available for ANOVA analysis.")

    # Step 10: Selecting Final Predictors for Building Machine Learning Model
    st.write("## Step 10: Selecting Final Predictors")
    # Ensure that numeric columns are selected
    selected_features = st.multiselect("Select predictor variables (independent variables):",
                                       data_cleaned.select_dtypes(include=['float64', 'int64']).columns)
    st.write("### Selected Features:", selected_features)

    # Check if the user selected features and the target variable
    if selected_features:
        st.write(f"### Target Variable: `{target}`")

        # Step 11: Data Preparation for Machine Learning
        st.write("## Step 11: Data Preparation for Machine Learning")
        # Extracting the features and target variable
        X = data_cleaned[selected_features]
        y = data_cleaned[target]

        # Splitting the data into train and test sets
        test_size = st.slider("Select the test size (percentage)", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Cache the scaler and scaled data
        scaler, X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        st.write(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Step 12: Model Training and Evaluation
        st.write("## Step 12: Model Training and Evaluation")
        # Check if models are already cached, otherwise train and cache them
        if 'trained_models' not in st.session_state:
            st.session_state.trained_models = train_models(X_train_scaled, y_train)

        trained_models = st.session_state.trained_models

        model_performance = {}
        for name, model in trained_models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # Store the results in the dictionary
            model_performance[name] = {"MSE": mse, "R2 Score": r2, "MAE": mae}

        # Convert the performance dictionary to a pandas DataFrame for better visualization
        performance_df = pd.DataFrame(model_performance).T  # Transpose to get model names as rows

        # Apply styles using pandas built-in styling
        styled_df = performance_df.style.format(precision=2) \
            .background_gradient(subset=["MSE"], cmap="Blues", low=0, high=1) \
            .background_gradient(subset=["R2 Score"], cmap="Greens", low=0, high=1) \
            .background_gradient(subset=["MAE"], cmap="Reds", low=0, high=1) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{
            'selector': 'th',
            'props': [('font-size', '14px'), ('text-align', 'center'), ('color', '#ffffff'),
                      ('background-color', '#404040')]
        }])

        # Display the table with st.table or st.dataframe
        st.write("## Model Performance Table")
        st.dataframe(styled_df, use_container_width=True)

        # Step 12: Visualizing the Performance Comparison between Models
        st.write("## Visualizing Model Performance Comparison")

        # Extracting model names and their respective performance metrics
        model_names = list(model_performance.keys())
        mse_values = [model_performance[model]["MSE"] for model in model_names]
        r2_values = [model_performance[model]["R2 Score"] for model in model_names]
        mae_values = [model_performance[model]["MAE"] for model in model_names]

        # Creating a bar plot to compare MSE, R2, and MAE across models
        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        # MSE Comparison
        ax[0].bar(model_names, mse_values, color='blue')
        ax[0].set_title("Model Comparison: Mean Squared Error (MSE)")
        ax[0].set_ylabel("MSE")

        # R2 Score Comparison
        ax[1].bar(model_names, r2_values, color='green')
        ax[1].set_title("Model Comparison: R2 Score")
        ax[1].set_ylabel("R2 Score")

        # MAE Comparison
        ax[2].bar(model_names, mae_values, color='red')
        ax[2].set_title("Model Comparison: Mean Absolute Error (MAE)")
        ax[2].set_ylabel("MAE")

        # Display the plot
        plt.tight_layout()
        st.pyplot(fig)

        # Step 13: Selecting the Best Model
        st.write("## Step 13: Selecting the Best Model")

        # Check if model performance dictionary has been populated
        if model_performance:
            # Select the model with the lowest MSE
            best_model_mse = min(model_performance, key=lambda x: model_performance[x]["MSE"])
            st.write(f"### Best Model based on Lowest Mean Squared Error (MSE): {best_model_mse}")

            # Step 14: Retraining the Best Model on Entire Data
            st.write("## Step 14: Retraining the Best Model")
            best_model = trained_models[best_model_mse]
            # Retrain the best model on the entire dataset
            # Combine and scale the entire dataset
            X_combined_scaled = scaler.fit_transform(X)
            best_model.fit(X_combined_scaled, y)

            # Save the best model in session state and also as a file
            if 'best_model' not in st.session_state:
                st.session_state.best_model = best_model

            # save the model after retraining
            model_filename = "best_model.pkl"
            joblib.dump(best_model, model_filename)
            st.write(f"Model `{best_model_mse}` has been retrained and saved as `{model_filename}`.")
        else:
            st.write("No model performance results available. Please ensure models were trained successfully.")

        # Step 15: Model Deployment - Load the Saved Model and Predict
        st.write("## Step 15: Model Deployment - Predict Using Saved Model")

        # Load the saved model
        model_filename = "best_model.pkl"

        try:
            loaded_model = joblib.load(model_filename)
            st.write(f"Model `{model_filename}` loaded successfully!")

            # Allow user to input values for the features
            st.write("### Provide the input values for prediction")

            # Generate input fields dynamically based on the selected features
            user_input_values = {}
            for feature in selected_features:  # Ensure selected_features from step 11 is available
                user_input_values[feature] = st.number_input(f"Enter value for {feature}",
                                                             value=float(data_cleaned[feature].mean()))

            # Add Predict button
            if st.button("Predict"):
                # Convert the user inputs into a DataFrame
                user_input_df = pd.DataFrame([user_input_values])

                # Scale the user inputs using the same scaler
                user_input_scaled = scaler.transform(user_input_df)

                # Make predictions using the loaded model
                predicted_value = loaded_model.predict(user_input_scaled)

                # Display the predicted value
                st.write(f"### Predicted {target}: {predicted_value[0]:.2f}")

                # Visualizing the Prediction with Input Features
                st.write("## Visualizing Prediction and Feature Values")

                # Create a combined bar plot for input features and prediction
                fig, ax = plt.subplots()

                # Plot the input feature values
                feature_names = list(user_input_values.keys())
                feature_values = list(user_input_values.values())

                ax.bar(feature_names, feature_values, color='lightblue', label='Feature Values')

                # Add the predicted value at the bottom of the chart
                ax.bar(['Predicted ' + target], [predicted_value[0]], color='orange', label='Predicted Value')

                ax.set_xlabel("Value")
                ax.set_title(f"Input Features and Predicted {target}")
                ax.legend()

                # Display the plot
                st.pyplot(fig)

                # Create a line chart for input features and predicted value
                fig, ax = plt.subplots(figsize=(10, 6))

                copy_feature_names = feature_names.copy()
                copy_feature_values = feature_values.copy()
                # Add the predicted value at the end
                copy_feature_names.append(f"Predicted {target}")
                copy_feature_values.append(predicted_value[0])

                ax.plot(copy_feature_names, copy_feature_values, marker='o', linestyle='-', color='blue',
                        label='Feature and Predicted Values')

                ax.set_xlabel("Features and Prediction")
                ax.set_ylabel("Values")
                ax.set_title(f"Input Features and Predicted {target}")
                ax.grid(True)

                # Add data labels
                for i, txt in enumerate(copy_feature_values):
                    ax.annotate(f'{txt:.2f}', (copy_feature_names[i], copy_feature_values[i]),
                                textcoords="offset points",
                                xytext=(0, 10), ha='center')

                # Display the plot
                st.pyplot(fig)

        except FileNotFoundError:
            st.write(f"Model `{model_filename}` not found. Please ensure the model has been saved correctly.")
    else:
        st.write("No numeric features selected for training.")
else:
    st.write("Please upload one or more CSV files from the sidebar to get started.")
