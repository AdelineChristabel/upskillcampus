#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf
import keras
 
#read the csv file
df = pd.read_csv(r'C:\Users\ADELINE CHRISTABEL\Desktop\CropProductionDataset.csv')

#df.shape

df.info()

df.describe()

df.head()
#Check for null values
df.isnull().sum()
#if null exist, drop them
df = df.dropna()
#drop the duplicates,if present
df = df.drop_duplicates()

#df.columns

df['Crop'].value_counts()

areas_of_production =df.groupby(['State'])['Cost of Production (`/Quintal) C2'].max().reset_index()
#areas_of_production

area = areas_of_production['State']
price = areas_of_production['Cost of Production (`/Quintal) C2']

fig = plt.figure(figsize = (19, 7))

# creating the bar plot
plt.bar(area, price, color ='magenta',
        width = 0.4)

plt.xlabel("STATE")
plt.ylabel("PRICE")
plt.title("Cost of Production  vs  State")
plt.show()

areas_of_production =df.groupby(['State'])['Crop'].max().reset_index()
#areas_of_production

col = [
    'Cost of Cultivation (`/Hectare) A2+FL', 
    'Cost of Cultivation (`/Hectare) C2', 
    'Cost of Production (`/Quintal) C2', 
    'Yield (Quintal/ Hectare) '
]

# Histograms
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, column in enumerate(col):
    sns.histplot(df[column], kde=True, ax=axes[i], color='blue')
    axes[i].set_title(column)
plt.tight_layout()
plt.show()

# Boxplots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, column in enumerate(col):
    sns.boxplot(data=df, x=column, ax=axes[i], color='green')
    axes[i].set_title(column)
plt.tight_layout()
plt.show()


# Correlation heatmap for numerical columns
corr_data = df[col].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, cmap='coolwarm', vmin=-1, vmax=1, square=True, linewidths=0.5)
plt.title('Heatmap of Correlations', fontsize=16)
plt.show()


# Preprocessing
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from sklearn.preprocessing import LabelEncoder
# Initialize separate label encoders for 'Crop' and 'State'
crop_encoder = LabelEncoder()
state_encoder = LabelEncoder()

# Fit the encoders to the respective columns
df['Crop'] = crop_encoder.fit_transform(df['Crop'])
df['State'] = state_encoder.fit_transform(df['State'])

# Save the encoders
joblib.dump(crop_encoder, 'Crop_encoder.pkl')
joblib.dump(state_encoder, 'State_encoder.pkl')
# Standardize numerical columns
numerical_columns = [
    'Cost of Cultivation (`/Hectare) A2+FL', 
    'Cost of Cultivation (`/Hectare) C2', 
    'Cost of Production (`/Quintal) C2', 
    'Yield (Quintal/ Hectare) '
]
# Scale numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
print(df.head())

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

df.columns = df.columns.str.strip()
# Check column names
print(df.columns)

# Splitting the dataset
from sklearn.model_selection import train_test_split
#print(df.columns)
# Correctly define the target column after ensuring the name matches
features = [
    'Crop', 
    'State', 
    'Cost of Cultivation (`/Hectare) A2+FL', 
    'Cost of Cultivation (`/Hectare) C2', 
    'Cost of Production (`/Quintal) C2'
]
target = 'Yield (Quintal/ Hectare)'  # Ensure this is the exact name
if target not in df.columns:
    raise ValueError(f"Column '{target}' is not found in the DataFrame.")

# Extract the features and target
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

df.to_csv('CropProductionTesting.csv', index=None)


# Model training and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Support Vector Machine": SVR(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
}


results = []
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({"Model": model_name, "MSE": mse, "R2": r2})

results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
print(results_df)

# Save the best model
best_model_name = results_df.iloc[0]["Model"]
print(f"Best model: {best_model_name}")

best_model = models[best_model_name]
best_model.fit(X_train, y_train)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Evaluate the models (example for one model)
y_pred = best_model.predict(X_test)  # Replace `best_model` with your model variable
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

import joblib

model_filename = f"{best_model_name}_CropProduction_model.pkl"
joblib.dump(best_model, model_filename)
#joblib.dump(le, 'l_encoder.pkl')
print(f"Model saved successfully as {model_filename}")


#-------------------------------------------------Streamlit framework--------------------------------
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_filename = 'Random Forest_CropProduction_model.pkl'  # Replace with your saved model filename
loaded_model = joblib.load(model_filename)

# Load the label encoders for categorical columns
crop_encoder = joblib.load('Crop_encoder.pkl')  # Replace with your saved crop encoder filename
state_encoder = joblib.load('State_encoder.pkl')  # Replace with your saved state encoder filename
scaler = joblib.load('scaler.pkl')


# Title of the app
st.title("Yield Prediction Application")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Preprocess input data
        #data['Crop'] = crop_encoder.transform(data['Crop'])
       # data['State'] = state_encoder.transform(data['State'])
        data[numerical_columns] = scaler.transform(data[numerical_columns])

        # Ensure the uploaded file contains the required columns
        required_columns = [
            'Crop', 'State', 
            'Cost of Cultivation (`/Hectare) A2+FL', 
            'Cost of Cultivation (`/Hectare) C2', 
            'Cost of Production (`/Quintal) C2'
        ]
        if not all(col in data.columns for col in required_columns):
            st.error("The uploaded file does not contain the required columns!")
        else:
            # Dropdown to select a row for prediction
            selected_index = st.selectbox("Select a row for prediction:", data.index)
            
            # Fetch row data for the selected index
            selected_row = data.loc[selected_index]

            # Prepare input data with dynamic values based on the selected row
            input_data = {}
            for col in required_columns:
                if col in ['Crop', 'State']:
                    # Dropdown for categorical inputs
                    unique_values = data[col].unique().tolist()
                    selected_value = st.selectbox(f"Select {col}:", unique_values, index=unique_values.index(selected_row[col]), key=col)
                    input_data[col] = crop_encoder.transform([selected_value])[0] if col == 'Crop' else state_encoder.transform([selected_value])[0]
                else:
                    # Number input for numerical values
                    input_data[col] = st.number_input(f"{col}:", 
                                                      value=float(selected_row[col]), 
                                                      step=0.0001, 
                                                      format="%.4f", 
                                                      key=col)

            # Convert input data to DataFrame for prediction
            input_df = pd.DataFrame([input_data])

            if st.button("Predict"):
                # Make prediction (scaled value)
                scaled_prediction = loaded_model.predict(input_df)[0]

                # Prepare a dummy row for inverse transformation
                # Assuming the scaler was trained with all features and the target column at the end
                dummy_row = [0] * len(numerical_columns)  # Create a list of zeros for all numerical columns
                dummy_row[-1] = scaled_prediction  # Place the prediction in the last position

                # Convert scaled prediction back to original value
                try:
                    original_prediction = scaler.inverse_transform([dummy_row])[:, -1][0]  # Get the last column (target)
                except Exception as e:
                    st.error(f"Error during inverse transformation: {e}")
                    original_prediction = scaled_prediction  # Fallback to scaled value

                # Decode categorical columns for display
                decoded_crop = crop_encoder.inverse_transform([int(input_data['Crop'])])[0]
                decoded_state = state_encoder.inverse_transform([int(input_data['State'])])[0]

                # Display results
                st.write(f"Yield Prediction (Scaled): {scaled_prediction:.4f}")
                st.write(f"Yield Prediction (Original): {original_prediction:.4f}")
                st.write(f"Crop: {decoded_crop}")
                st.write(f"State: {decoded_state}")


    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.write("Please upload a CSV file for prediction or manually select inputs.")




