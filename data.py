import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# File path for the dataset
file_path = "E:/mentalHealth/mental_health_dataset.csv"  # Adjust this to the actual path of your CSV file
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}. Please provide the correct file path.")
    exit()

# Load the dataset
columns = [
    "Timestamp", "Gender", "Country", "Occupation", "self_employed", "family_history", "treatment",
    "Days_Indoors", "Growing_Stress", "Changes_Habits", "Mental_Health_History", "Mood_Swings",
    "Coping_Struggles", "Work_Interest", "Social_Weakness", "mental_health_interview", "care_options"
]
data = pd.read_csv(file_path, names=columns, skiprows=1)

# Drop the 'Timestamp' column as it's not relevant
data.drop(columns=["Timestamp"], inplace=True)

# Handle missing values
data.fillna({
    'Gender': 'Unknown',
    'Country': 'Unknown',
    'Occupation': 'Unknown',
    'self_employed': 'No',
    'family_history': 'No',
    'treatment': 'No',
    'Days_Indoors': '1-14 days',  # Default to '1-14 days' if missing
    'Growing_Stress': 'No',
    'Changes_Habits': 'No',
    'Mental_Health_History': 'No',
    'Mood_Swings': 'No',
    'Coping_Struggles': 'No',
    'Work_Interest': 'Medium',  # Default to 'Medium' if missing
    'Social_Weakness': 'No',
    'mental_health_interview': 'No',
    'care_options': 'No'
}, inplace=True)

# Preprocess 'Days_Indoors' to convert to numeric values
def preprocess_days(value):
    if '1-14 days' in value:
        return 7  # Approximate average
    elif '15-30 days' in value:
        return 22  # Approximate average
    return 0  # Default for unrecognized values

data['Days_Indoors'] = data['Days_Indoors'].apply(preprocess_days)

# Encode binary categorical variables as 0 and 1
binary_columns = ['Growing_Stress', 'Changes_Habits', 'Mental_Health_History', 
                  'Mood_Swings', 'Coping_Struggles', 'Social_Weakness', 
                  'mental_health_interview', 'care_options', 'self_employed', 'family_history', 'treatment']
for col in binary_columns:
    data[col] = data[col].map({'Yes': 1, 'No': 0})

# One-hot encode categorical variables with multiple categories
data = pd.get_dummies(data, columns=['Gender', 'Country', 'Occupation', 'Work_Interest'], drop_first=True)

# Define features (X) and target (y)
X = data.drop(columns=['treatment'])
y = data['treatment']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model to a .pkl file
output_model_path = "random_forest_model.pkl"
with open(output_model_path, 'wb') as file:
    pickle.dump(rf_model, file)

print(f"Random Forest model saved as '{output_model_path}'")