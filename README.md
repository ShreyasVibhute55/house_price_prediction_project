#  House Price Prediction ML Project

This project is an end-to-end machine learning pipeline that predicts **house prices** based on features like area, number of bedrooms, bathrooms, parking, and furnishing status. It includes data preprocessing, model training, performance evaluation, and a **Streamlit web application** to interactively use the model.

---

##  Problem Statement

Accurately predicting house prices based on property features is essential for buyers, sellers, and real estate companies. This project aims to build a regression model using a real estate dataset to predict house prices in a city.

---

##  Folder Structure
HOUSE_PRICE_PREDICTION_PROJECT/
│
├── data/ # Contains raw Housing.csv dataset
├── models/ # Trained and saved ML model (house_model.pkl)
├── app/ # Streamlit app code
│ └── app.py
├── scripts/ # Python scripts for model training
│ └── train_model.py
├── README.md # Documentation file
└── requirements.txt # All Python dependencies

---

##  Dataset Information

- **File Name:** Housing.csv
- **Columns/Features:**
  - `area`
  - `bedrooms`
  - `bathrooms`
  - `stories`
  - `parking`
  - `furnishingstatus`
  - `price` (Target variable)

---

##  How to Use This Project

### Step 1: Clone this Repository

```bash
git clone https://github.com/your-username/HOUSE_PRICE_PREDICTION_PROJECT.git
cd HOUSE_PRICE_PREDICTION_PROJECT

Step 2: Set Up Virtual Environment (Optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows

Step 3: Install Requirements
pip install -r requirements.txt

Step 4: Train the Model
cd scripts
python train_model.py
• Model will be trained
• Saved as models/house_model.pkl

Step 5: Run Streamlit Web App
cd app
streamlit run app.py
• The app will open in your browser
• Input property features to get predicted price

# Model Performance
Metric	Score
R² Score	0.6529
MAE	₹970,043.40
RMSE	₹1,324,506.96

# Interpreted:

The model explains ~65% variance in the data.

Mean error is around ₹9.7 Lakhs.

RMSE shows standard deviation of prediction errors (~₹13.24 Lakhs).

# Technologies Used
Python 3.10+

Pandas, NumPy

Scikit-learn

Joblib

Streamlit

Matplotlib / Seaborn (for EDA, optional)

# Python Requirements
Install all packages via:pip install -r requirements.txt
Example requirements.txt content:pandas
numpy
scikit-learn
streamlit
joblib

# About the Author
• Shreyas Vibhute
• Pune, India
• Data Science & ML Enthusiast
• Learning AI, ML, and Real-Time Project Development