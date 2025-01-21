# 🌾 Crop Yield Prediction System 🌱

A data-driven solution to predict crop yield based on attributes such as cost of cultivation, state, crop, cost of production, and yield. This application leverages machine learning techniques to provide actionable insights, aiding farmers and policymakers in optimizing agricultural practices.

---

## 📋 Table of Contents

1. [✨ Features](#-features)
2. [🚀 Technologies Used](#-technologies-used)
3. [📂 Dataset](#-dataset)
4. [⚡ Getting Started](#-getting-started)
5. [🎯 Usage](#-usage)
6. [📊 Model Training and Evaluation](#-model-training-and-evaluation)
7. [📈 Results](#-results)
8. [🚧 Future Enhancements](#-future-enhancements)
9. [🙏 Acknowledgments](#-acknowledgments)
10. [📜 License](#-license)

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ✨ Features

🌟 **Accurate Predictions**: Predicts crop yield based on input attributes like cost of cultivation, state, and crop.

📁 **Flexible Data Input**: Accepts CSV file uploads or manual data entry for predictions.

📊 **Data Visualizations**: Includes detailed plots such as histograms, scatter plots, and correlation matrices to analyze relationships and trends.

⚙️ **Robust Model Selection**: Compares multiple machine learning models to select the best-performing one.

💾 **Model Persistence**: Saves trained models and encoders for seamless reuse.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Technologies Used

- **Programming Language**: Python
- **Machine Learning**: Scikit-learn
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Web Framework**: Streamlit
- **Model Persistence**: Joblib

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 Dataset

The dataset used contains the following features:

- **Crop**: Type of crop (e.g., wheat, paddy, maize etc.).
- **State**: Location where the crop is cultivated.
- **Cost of Cultivation (`/Hectare) A2+FL**: The total cost incurred per hectare using the A2+FL methodology.
- **Cost of Cultivation (`/Hectare) C2**: The total cost incurred per hectare using the C2 methodology.
- **Cost of Production (`/Quintal) C2**: The cost incurred per quintal of crop production using the C2 methodology.
- **Yield (Quintal/ Hectare)**: The yield in quintals per hectare.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## ⚡ Getting Started

### Prerequisites

- Python 3.8 or higher.
- Required libraries listed in `requirements.txt`.

### Installation

1. Clone the repository:
   ```bash
   git clone  https://github.com/AdelineChristabel/upskillcampus.git
   ```

2. Navigate to the directory:
   ```bash
   cd upskillcampus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🎯 Usage

### Step 1: Train the Model

Run the script to preprocess the dataset, train the model, and save it:
```bash
python CropYieldPrediction.py
```
This generates the following files:

- **Random Forest_CropProduction_model.pkl**: The trained model.
- **Crop_encoder.pkl**: The label encoder for categorical variable Crop.
- **State_encoder.pkl**: The label encoder for categorical variable State.
- **Scaler.pkl**: The scaler for numerical variables.
### Step 2: Launch the Web Application

Start the Streamlit app:
```bash
streamlit run CropYieldPrediction.py
```

### Step 3: Predict Crop Yield

1. **Upload a CSV file** with columns: Crop, State, Cost of Cultivation A2+FL, Cost of Cultivation C2, Cost of Production C2, and Yield.
2. **Manually input values** in the provided fields.
3. View the predicted yield and visualizations of input data.
------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 📊 Model Training and Evaluation

The following machine learning models were evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)
- K-Nearest Neighbors (KNN)

The **Random Forest Regressor** was selected based on its high R² score and low Mean Squared Error (MSE).

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📈 Results

The model achieved the following performance metrics on the test dataset:

- **R² Score**: 0.928498
- **Mean Squared Error (MSE)**: 0.109590

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚧 Future Enhancements

🌐 **Multi-language Support**: Enable predictions in multiple languages.

📊 **Advanced Visualizations**: Integrate interactive visualizations for enhanced insights.

📡 **Real-time Data Integration**: Incorporate live weather and market data for dynamic predictions.

🎛️ **Customizable Inputs**: Allow users to assign weights to various parameters for personalized predictions.

------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🙏 Acknowledgments

Special thanks to the contributors and maintainers of the dataset and to the developers of the libraries and tools utilized in this project.

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

