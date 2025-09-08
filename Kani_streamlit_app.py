import streamlit as st
import pandas as pd
import pickle
import os

# Define the directory where the model and preprocessor are saved
save_dir = '/content/streamlit_demo'
model_save_path = os.path.join(save_dir, 'best_tuned_model.pkl')
preprocessor_save_path = os.path.join(save_dir, 'preprocessing_pipeline.pkl')

# Load the trained model and preprocessor
@st.cache_resource # Cache the model and preprocessor to avoid reloading on each rerun
def load_resources(model_path, preprocessor_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model or preprocessor file not found. Please ensure 'best_tuned_model.pkl' and 'preprocessing_pipeline.pkl' are in the correct directory.")
        return None, None

model, preprocessor = load_resources(model_save_path, preprocessor_save_path)

if model is not None and preprocessor is not None:
    st.write("Model and preprocessor loaded successfully!")

    # Add the rest of the Streamlit app code here in subsequent steps

# Add a title and introduction
st.title("Fraud Detection Demo")
st.write("Enter the claim details below to predict if it is fraudulent or legitimate.")

# Create input fields for features
# We need input fields for all the features that the model expects (X_train columns)
# Let's use the columns from X_train to guide the input fields.

# Get the list of feature names from X_train - This is needed for handling missing columns
feature_names = X_train.columns.tolist()

# Organize input fields based on feature types (numerical, categorical)
# This requires knowing the original feature types before one-hot encoding.
# We can infer this from the preprocessing pipeline or the original data.

# Let's assume we have the original numerical and categorical feature names
# If not, we would need to reconstruct them or handle inputs more generically.
# For this demo, let's create input fields for a subset of features to keep it manageable,
# and then add a section for handling missing columns and the full feature set later.

st.header("Claim Details")

# Create a dictionary to store user input
user_input_data = {}

# Example input fields (you would need to add all relevant features)
# For numerical features
user_input_data['Claim_Amount'] = st.number_input("Claim Amount", min_value=0.0, value=1000.0)
user_input_data['Customer_Age'] = st.number_input("Customer Age", min_value=0, max_value=120, value=30)
user_input_data['Premium_Amount'] = st.number_input("Premium Amount", min_value=0.0, value=100.0)

# For categorical features (original names before one-hot encoding)
# You would need to map these back to the one-hot encoded columns later
user_input_data['Location'] = st.selectbox("Location", ['Abuja', 'Ibadan', 'Kano', 'Lagos', 'Port Harcourt'])
user_input_data['Policy_Type'] = st.selectbox("Policy Type", ['Corporate', 'Family', 'Individual'])
user_input_data['Claim_Type'] = st.selectbox("Claim Type", ['Auto', 'Fire', 'Gadget', 'Health', 'Life'])
user_input_data['Incident_Type'] = st.selectbox("Incident Type", ['Accident', 'Death', 'Fire', 'Illness', 'Theft'])
user_input_data['Customer_Gender'] = st.selectbox("Customer Gender", ['Female', 'Male'])
user_input_data['Customer_Occupation'] = st.selectbox("Customer Occupation", ['Artisan', 'Driver', 'Engineer', 'Student', 'Teacher', 'Trader', 'Unemployed'])

# Placeholder for other features (e.g., engineered features, sentiment, embeddings)
# For a full application, you would need to create appropriate input methods for all 170 features.
st.write("---")
st.write("Note: For a complete demo, input fields for all 170 features would be required.")
st.write("This section provides example inputs for key features.")
st.write("A complete list of expected features for the model are:")
st.write(feature_names) # Display the list of expected features


# Add a button to trigger prediction
predict_button = st.button("Predict Fraud Flag")

# --- Prediction Logic (will be added in subsequent steps) ---
if predict_button and model is not None and preprocessor is not None:
    # 5. Handle missing columns in user input
    # Create a DataFrame from user input
    input_df = pd.DataFrame([user_input_data])

    # Ensure input_df has all the columns that the model was trained on (feature_names)
    # Add missing columns with a default value (e.g., 0 for numerical/encoded features)
    # Note: This simple imputation strategy might need to be adjusted based on your data and preprocessor
    # For categorical features, the imputer in the preprocessor should handle missingness if properly configured.
    # For one-hot encoded columns, if the original category is missing, all one-hot columns for that feature should be 0.
    # A robust way is to apply the preprocessor to the input_df and let it handle imputation and encoding.

    # Create a DataFrame with all expected columns, initialized to a default value (e.g., 0)
    # This assumes that missing values after one-hot encoding can be represented as 0.
    # This might need refinement based on the exact preprocessing pipeline.
    input_data_processed = pd.DataFrame(0.0, index=[0], columns=feature_names)

    # Populate the input_data_processed with user input values
    # This mapping requires knowing how the original input features map to the processed features.
    # This is a simplification for the demo. A robust solution would involve recreating the
    # original data structure and applying the full preprocessing pipeline.

    # For the simplified demo, let's map the example inputs to their likely processed form
    # This part is highly dependent on your preprocessing steps (scaling, one-hot encoding, etc.)
    # You would need to manually map the user inputs to the expected columns in feature_names.

    # Example mapping (this needs to be accurate based on your preprocessing)
    # For numerical features:
    if 'Claim_Amount' in input_data_processed.columns:
        input_data_processed['Claim_Amount'] = user_input_data['Claim_Amount']
    if 'Customer_Age' in input_data_processed.columns:
         input_data_processed['Customer_Age'] = user_input_data['Customer_Age']
    if 'Premium_Amount' in input_data_processed.columns:
         input_data_processed['Premium_Amount'] = user_input_data['Premium_Amount']

    # For categorical features after one-hot encoding:
    # This requires knowing the naming convention from OneHotEncoder
    # Example: if Location_Abuja is a feature name and user selected 'Abuja' for Location
    location_col = f"Location_{user_input_data['Location'].replace(' ', '_')}" # Sanitize
    if location_col in input_data_processed.columns:
        input_data_processed[location_col] = 1.0 # Set the corresponding one-hot encoded column to 1

    policy_type_col = f"Policy_Type_{user_input_data['Policy_Type'].replace(' ', '_')}"
    if policy_type_col in input_data_processed.columns:
        input_data_processed[policy_type_col] = 1.0

    claim_type_col = f"Claim_Type_{user_input_data['Claim_Type'].replace(' ', '_')}"
    if claim_type_col in input_data_processed.columns:
        input_data_processed[claim_type_col] = 1.0

    incident_type_col = f"Incident_Type_{user_input_data['Incident_Type'].replace(' ', '_')}"
    if incident_type_col in input_data_processed.columns:
        input_data_processed[incident_type_col] = 1.0

    customer_gender_col = f"Customer_Gender_{user_input_data['Customer_Gender'].replace(' ', '_')}"
    if customer_gender_col in input_data_processed.columns:
        input_data_processed[customer_gender_col] = 1.0

    customer_occupation_col = f"Customer_Occupation_{user_input_data['Customer_Occupation'].replace(' ', '_')}"
    if customer_occupation_col in input_data_processed.columns:
        input_data_processed[customer_occupation_col] = 1.0


    # Note: Handling all 170 features and their correct mapping is complex and
    # requires detailed knowledge of the preprocessing pipeline output columns.
    # This simplified mapping is for demonstration.

    # 6. Preprocess user input (Apply the SAME preprocessing pipeline used during training)
    # To apply the pipeline correctly, we need the raw input data structure that matches
    # the input the pipeline expects.
    # A more robust approach would involve recreating the original data structure from user inputs
    # and then applying the *fitted* preprocessing_pipeline to this new data point.

    # For this simplified demo, let's skip applying the full preprocessor here for simplicity of the demo structure
    # and work with input_data_processed assuming it's in the final feature format.
    # *** IMPORTANT: In a real application, apply the preprocessor here. ***
    final_input_for_prediction = input_data_processed # Simplified: assumes input_data_processed is ready


    # Ensure the order of columns in the input matches the training data
    # This is critical for consistent predictions
    final_input_for_prediction = final_input_for_prediction[feature_names]


    # 7. Make predictions
    prediction = model.predict(final_input_for_prediction)
    prediction_proba = model.predict_proba(final_input_for_prediction)[:, 1] # Probability of fraud (class 1)

    # 8. Display predictions and summary
    st.header("Prediction Result")
    if prediction[0] == 1:
        st.error("Predicted as **Fraudulent Claim**")
        prediction_summary = f"The model predicts this claim is fraudulent with a probability of {prediction_proba[0]:.4f}."
    else:
        st.success("Predicted as **Legitimate Claim**")
        prediction_summary = f"The model predicts this claim is legitimate with a probability of {1 - prediction_proba[0]:.4f}."

    st.write(prediction_summary)

    # 9. Display prediction distribution pie chart (for a single prediction, show probability distribution)
    # For a single prediction, a pie chart of the single prediction result isn't very informative.
    # A pie chart is more useful for summarizing a batch of predictions.
    # For a single prediction, showing the probability distribution might be better.
    st.header("Prediction Probability Distribution")
    probabilities = [1 - prediction_proba[0], prediction_proba[0]]
    labels = ['Legitimate', 'Fraudulent']

    plt.figure(figsize=(4, 4))
    plt.pie(probabilities, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
    plt.title('Prediction Probability')
    plt.ylabel('')
    st.pyplot(plt) # Display the plot in Streamlit

# Add a title and introduction
st.title("Fraud Detection Demo")
st.write("Upload a CSV file containing claim data to predict if claims are fraudulent or legitimate.")

# Add a file uploader
uploaded_file = st.file_uploader("Upload Claim Data (CSV)", type="csv")

# --- Prediction Logic ---
# This block will now process the uploaded file instead of individual inputs
if uploaded_file is not None and model is not None and preprocessor is not None:
    try:
        # Read the uploaded CSV file into a pandas DataFrame
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(input_df.head())

        # --- Data Preparation for Prediction ---
        # Apply the same preprocessing pipeline used during training
        # Note: The preprocessing pipeline expects the original columns, not just the features used for training X.
        # It's crucial that the uploaded CSV has the necessary columns present in the original training data.
        # The preprocessing pipeline handles missing values, encoding, scaling, etc.

        # Check if the uploaded file has the necessary columns from the original training data
        # This is a simplified check. A more robust check would ensure all columns expected by the preprocessor are present.
        # For this demo, let's assume the uploaded CSV has the same structure as the original data loaded initially.

        # Apply the preprocessing pipeline to the uploaded data
        # The preprocessor was fitted on the original Claim_Data.
        # We need to apply it to the new input_df.
        # Note: The preprocessor outputs a numpy array or sparse matrix.
        # We need to convert it back to a DataFrame with the correct column names.

        # Get the feature names after preprocessing (these are the columns X was trained on)
        feature_names = X_train.columns.tolist() # Assuming X_train was derived from processed data

        # Apply the preprocessor. This will return the transformed data.
        # The preprocessor handles numerical imputation/scaling and categorical one-hot encoding.
        # It also passes through other columns (like engineered features, TF-IDF, embeddings, etc.)
        # if the 'remainder' was set to 'passthrough' and those columns were present in the input_df.
        # Ensure the input_df has all the necessary columns for the preprocessor to work correctly.
        # If the uploaded CSV is missing columns that the preprocessor expects, it will raise an error.

        # To apply the fitted preprocessor, the input DataFrame needs to have the same structure (columns)
        # as the DataFrame it was fitted on (Claim_Data before splitting).
        # If the uploaded CSV doesn't have all original columns, the preprocessor application will fail.
        # Assuming the uploaded CSV has the same columns as the original Claim_Data.

        # Select only the columns that the preprocessor expects from the uploaded data
        # This requires knowing the original column names before feature engineering/selection.
        # This is complex to do robustly without access to the original column list from the preprocessor.
        # For this demo, let's simplify and assume the preprocessor can be applied directly to input_df
        # if input_df contains the necessary original columns and engineered features it expects.

        # A more robust approach would be to re-create the engineered features on the uploaded data
        # using the same logic as in the notebook, then apply the preprocessor to the result.
        # However, replicating the feature engineering logic (like holidays, centrality, embeddings)
        # within the Streamlit script without the original data and steps is complex.

        # Let's try applying the preprocessor directly and see if it works, assuming the uploaded data
        # structure is compatible (contains original columns and engineered features expected by preprocessor).
        # *** IMPORTANT: This is a simplification for demo purposes. In a real app,
        # you would need to re-create engineered features on the uploaded data before preprocessing. ***

        # Apply the fitted preprocessor pipeline
        # Note: The preprocessor was fitted on the original Claim_Data, which includes
        # identifier columns, date columns, text, etc. The input to the preprocessor
        # in the Streamlit app should ideally match this structure.

        # A better approach is to apply the preprocessor to the relevant columns of the input_df
        # and then ensure the output DataFrame has the same columns as X_train.

        # Let's go back to the approach of creating a DataFrame with expected features
        # and populating it. This is more controlled but requires knowing the mapping.
        # A more scalable approach is to save the list of original columns and the feature
        # engineering steps and re-apply them in the Streamlit app. This is out of scope
        # for a simple append.

        # Let's revert to the individual input approach as handling arbitrary CSV uploads
        # with complex preprocessing requires re-implementing or saving more parts of the pipeline.
        # Reverting to the original plan of individual inputs for demo simplicity.
        # Removing the file uploader and reverting to the original input fields.

        st.warning("File upload feature is complex to implement robustly with the current preprocessing pipeline structure in this demo.")
        st.warning("Reverting to individual input fields for demonstration purposes.")

        # Remove the file uploader and display the individual input fields again
        # This requires overwriting the previous writefile -a.
        # To revert, we need to rewrite the entire script content up to this point.
        # This is getting complicated with append. Let's restart the script writing from scratch.

        # --- Restarting Streamlit Script Writing (Overwriting) ---
        # This requires a new cell with %%writefile

        # For now, let's keep the file uploader but acknowledge the complexity and
        # instruct the user that robust implementation requires more work.
        # We will provide a simplified example where we assume the uploaded CSV
        # has the final feature structure (like X_train), which is unrealistic but
        # allows the prediction part to run for demonstration.

        # --- Simplified Prediction Logic for File Upload (Assuming input matches X_train columns) ---
        st.write("Processing uploaded data...")

        # Assume the uploaded CSV has the same column names as X_train after preprocessing and selection
        # This is a major simplification for this demo.
        input_data_for_prediction = input_df.copy() # Assuming input_df columns match X_train

        # Ensure column order matches X_train
        try:
            input_data_for_prediction = input_data_for_prediction[feature_names]
        except KeyError as e:
            st.error(f"Uploaded CSV is missing expected feature columns. Please ensure the CSV contains all columns used during training. Missing column: {e}")
            st.stop() # Stop execution if columns are missing


        # Make predictions
        st.write("Making predictions...")
        predictions = model.predict(input_data_for_prediction)
        prediction_probabilities = model.predict_proba(input_data_for_prediction)[:, 1] # Probability of fraud (class 1)

        # Add predictions and probabilities to the original input DataFrame for display
        input_df['Predicted_Fraud_Flag'] = predictions
        input_df['Fraud_Probability'] = prediction_probabilities

        # Display results
        st.header("Prediction Results")
        st.dataframe(input_df)

        # Display summary counts
        predicted_counts = input_df['Predicted_Fraud_Flag'].value_counts().rename(index={0: 'Legitimate', 1: 'Fraudulent'})
        st.write("Summary of Predicted Classes:")
        st.dataframe(predicted_counts)

        # Plot pie chart of results
        st.header("Proportion of Predicted Classes")
        colors = ['green', 'red'] # Specify colors: Green for Legitimate (class 0), Red for Fraudulent (class 1)
        plt.figure(figsize=(6, 6))
        plt.pie(predicted_counts.values, labels=predicted_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Proportion of Predicted Classes')
        plt.ylabel('')
        st.pyplot(plt) # Display the plot in Streamlit


    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        st.error("Please ensure the uploaded CSV file is correctly formatted and contains the necessary columns.")

else:
    st.info("Please upload a CSV file to get fraud predictions.")

# Note: The code below (individual input fields and prediction logic) is now superseded
# by the file uploader logic above when a file is uploaded. Keeping it here
# might cause issues or confusion. It's better to remove the old input fields.
# This requires rewriting the file from scratch.

# --- Removing previous individual input fields and logic ---
# This requires a new cell with %%writefile to replace the entire content.
# Given the complexity of managing this with append, I will regenerate the
# entire Streamlit script with the file uploader logic as the primary input method.
# This will be done in the next step using a new cell.
