
import streamlit as st
import pandas as pd
import joblib
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://static.vecteezy.com/system/resources/previews/002/853/553/large_2x/doctor-wearing-uniform-smiling-while-presenting-and-pointing-isolated-on-blue-background-with-copy-space-photo.jpg"); /* Replace with your image URL */
    background-size: cover;
    background-repeat: no-repeat;
    color: white; /* Makes text readable on darker backgrounds */
}
input, select, textarea {
    border: 5px solid black; /* Permanent black border */
    background-color: rgba(255, 255, 255, 0.8); /* Transparent white input box */
    color: #000000;
    padding: 10px;
    font-size: 16px;
    border-radius: 5px; /* Rounded corners for aesthetics */
}
.stButton>button {
    border: 2px solid black; /* Ensures buttons have black borders */
    background-color: rgba(255, 255, 255, 0.8); /* Button background color */
    color: #000000;
    padding: 10px;
    font-size: 16px;
    border-radius: 5px;
    font-weight: bold;
}
</style>
'''

# Apply CSS styling
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the trained Random Forest model
model_path = "E:/mentalHealth/random_forest_model.pkl"  # Adjust the path
try:
    rf_model = joblib.load(model_path)
    model_feature_names = rf_model.feature_names_in_  # Retrieve the feature names used in training
except FileNotFoundError:
    st.error(f"Model file not found at {model_path}. Please check the path.")
    st.stop()
except MemoryError:
    st.error("MemoryError: Unable to load the model file due to insufficient system memory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")
    st.stop()

# Define preprocessing for 'Days_Indoors'
def preprocess_days(value):
    if '1-14 days' in value:
        return 7  # Approximate average
    elif '15-30 days' in value:
        return 22  # Approximate average
    return 0  # Default for unrecognized values

# Streamlit app
st.markdown(
    '<div style="font-size:40px; font-weight:bold; color:black;">Mental Health Illness Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="font-size:16px; color:black;">Provide your information below to predict if mental health treatment might be recommended.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div style="font-size:20px; color:blue;">Enter Details :</div>',
    unsafe_allow_html=True
)


# User Input
gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
country = st.text_input("Country", "Enter your country")
occupation = st.text_input("Occupation", "Enter your occupation")
self_employed = st.radio("Self-employed?", ["Yes", "No"])
family_history = st.radio("Family history of mental health issues?", ["Yes", "No"])
days_indoors = st.selectbox("Days spent indoors", ["1-14 days", "15-30 days"])
growing_stress = st.radio("Growing stress?", ["Yes", "No"])
changes_habits = st.radio("Changes in habits?", ["Yes", "No"])
mental_health_history = st.radio("Mental health history?", ["Yes", "No"])
mood_swings = st.radio("Mood swings?", ["Yes", "No"])
coping_struggles = st.radio("Struggles with coping?", ["Yes", "No"])
work_interest = st.selectbox("Work interest level", ["Low", "Medium", "High"])
social_weakness = st.radio("Social weakness?", ["Yes", "No"])
mental_health_interview = st.radio("Had a mental health interview?", ["Yes", "No"])
care_options = st.radio("Available care options?", ["Yes", "No"])

# Preprocess Input Data
raw_input_data = {
    "Gender": gender,
    "Country": country,
    "Occupation": occupation,
    "self_employed": 1 if self_employed == "Yes" else 0,
    "family_history": 1 if family_history == "Yes" else 0,
    "Days_Indoors": preprocess_days(days_indoors),
    "Growing_Stress": 1 if growing_stress == "Yes" else 0,
    "Changes_Habits": 1 if changes_habits == "Yes" else 0,
    "Mental_Health_History": 1 if mental_health_history == "Yes" else 0,
    "Mood_Swings": 1 if mood_swings == "Yes" else 0,
    "Coping_Struggles": 1 if coping_struggles == "Yes" else 0,
    "Work_Interest": work_interest,
    "Social_Weakness": 1 if social_weakness == "Yes" else 0,
    "mental_health_interview": 1 if mental_health_interview == "Yes" else 0,
    "care_options": 1 if care_options == "Yes" else 0,
}

# Create a DataFrame
input_df = pd.DataFrame([raw_input_data])

# Ensure input matches the trained model's features
for feature in model_feature_names:
    if feature not in input_df.columns:
        input_df[feature] = 0  # Add missing features with default values
input_df = input_df[model_feature_names]  # Reorder columns to match the model's features

if st.button("Predict"):
    try:
        prediction = rf_model.predict(input_df)[0]
        prediction_proba = rf_model.predict_proba(input_df).tolist()

        # Prediction display with black text
        if prediction == 1:
            st.markdown('<p style="color:black; font-size:16px;">Prediction: <b>Treatment Recommended</b></p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:black; font-size:16px;">Prediction: <b>No Treatment Recommended</b></p>', unsafe_allow_html=True)
         
        if prediction == 1:
            st.markdown('<h3 style="color:black;">Diagnosis :</h3>', unsafe_allow_html=True)
            max_proba = max(prediction_proba[0])

            # Identify the most probable diagnosis
            if max_proba > 0.8:
                diagnosis = "**Major Depressive Disorder (MDD):** Persistent sadness or loss of interest affecting daily activities."
            elif 0.5 < max_proba <= 0.8:
                diagnosis = "**Generalized Anxiety Disorder (GAD):** Excessive worry or anxiety about various life events."
            else:
                diagnosis = "**Mild Stress-Related Symptoms:** Including mild anxiety or difficulty coping with changes."

            # Display diagnosis with black text
            st.markdown(f'<p style="color:black; font-size:16px;">{diagnosis}</p>', unsafe_allow_html=True)

            # Adding "Recommended Actions" with black color
            st.markdown('<h3 style="color:black;">Recommended Actions:</h3>', unsafe_allow_html=True)
            if "MDD" in diagnosis:
                st.markdown("""
                <ul style="color:black;">
                    <li><b>Psychotherapy:</b> CBT or Interpersonal Therapy (IPT).</li>
                    <li><b>Medication:</b> SSRIs (e.g., Escitalopram), SNRIs (e.g., Duloxetine).</li>
                    <li><b>Lifestyle:</b> Engage in regular physical activity and maintain a sleep routine.</li>
                </ul>
                """, unsafe_allow_html=True)
            elif "GAD" in diagnosis:
                st.markdown("""
                <ul style="color:black;">
                    <li><b>Therapy:</b> Cognitive Behavioral Therapy (CBT), Exposure Therapy.</li>
                    <li><b>Medication:</b> Short-term benzodiazepines or SSRIs (e.g., Sertraline).</li>
                    <li><b>Stress Management:</b> Incorporate mindfulness techniques and relaxation exercises.</li>
                </ul>
                """, unsafe_allow_html=True)
            else:  # Mild stress-related symptoms
                st.markdown("""
                <ul style="color:black;">
                    <li><b>Stress Reduction:</b> Focus on yoga, meditation, or guided relaxation techniques.</li>
                    <li><b>Healthy Living:</b> Regular exercise, balanced meals, and consistent sleep schedules.</li>
                </ul>
                """, unsafe_allow_html=True)

        else:
            st.subheader("General Wellness Tips:")
            st.markdown("""
            <ul style="color:black;">
                <li><b>Preventive Care:</b> Routine mental health checkups with professionals.</li>
                <li><b>Social Engagements:</b> Participate in community services, group activities, or hobbies.</li>
                <li><b>Healthy Lifestyle:</b> Maintain physical activity and a balanced diet.</li>
                <li><b>Mindfulness Practices:</b> Guided meditation or breathing exercises daily.</li>
                <li><b>Sleep Hygiene:</b> Develop and follow a regular sleep schedule.</li>
            </ul>
            """, unsafe_allow_html=True)
    except MemoryError:
          st.error("MemoryError: Unable to process prediction due to insufficient system memory.")
    except Exception as e:
     st.error(f"Error making prediction: {str(e)}")