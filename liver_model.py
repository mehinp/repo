
#LIVER MODEL

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


pd.set_option('future.no_silent_downcasting', True)

# File paths
train_data_path = '/Users/mehinhome/Documents/Mehin APL Project/Data/TRAIN Liver Data.csv'
test_data_path = '/Users/mehinhome/Documents/Mehin APL Project/Data/TEST LIVER Data.csv' 
drug_data_path='/Users/mehinhome/Documents/Mehin APL Project/Data/Beta Drug Data.csv'  # Path to the generated drug dataset

# Load datasets
train_data = pd.read_csv(train_data_path, encoding='latin1')
test_data = pd.read_csv(test_data_path)
drug_data = pd.read_csv(drug_data_path)


# Standardize column names in test_data
test_data.columns = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphatase",
    "ALT", "AST", "Total_Proteins", "Albumin", "Albumin_Globulin_Ratio"
]
test_data['Result'] = None

# Convert 'Gender' column to numeric (Male = 1, Female = 0)
train_data['Gender of the patient'] = train_data['Gender of the patient'].map({'Male': 1, 'Female': 0})
test_data['Gender'] = test_data['Gender'].map({'Male': 1, 'Female': 0})

# Handle missing values
train_data.fillna(train_data.mean(), inplace=True)
test_data = test_data.fillna(test_data.select_dtypes(include=[np.number]).mean()).infer_objects()



# Combine patient data
combined_patient_data = pd.concat([train_data, test_data], ignore_index=True)

# Add ALT/AST Ratio as a new feature
combined_patient_data['ALT/AST Ratio'] = (
    combined_patient_data[' Sgpt Alamine Aminotransferase'] / combined_patient_data['Sgot Aspartate Aminotransferase']
)

# Merge patient data with drug data (simulate drugs administered to patients)
# Match drug_data rows to patient_data rows by random assignment
combined_patient_data = combined_patient_data.sample(frac=1, random_state=42).reset_index(drop=True)
drug_data = drug_data.sample(frac=1, random_state=42).reset_index(drop=True)
merged_data = pd.concat([combined_patient_data, drug_data.iloc[:len(combined_patient_data)]], axis=1)

# Validate the merged data
print("Shape of merged data:", merged_data.shape)

# Select features and target variable
features = merged_data[[
    "Age of the patient", "Total_Bilirubin", "Direct_Bilirubin",
    " Alkphos Alkaline Phosphotase", "Sgot Aspartate Aminotransferase",
    "Total Protiens", " ALB Albumin", "A/G Ratio Albumin and Globulin Ratio", "ALT/AST Ratio",
    "Partition_Coefficient_LogP", "pKa", "Total_Body_Clearance_CLT",
    "Molecular_Weight", "Protein_Binding_Percent", "IC50_Sodium_Channel_Inhibition"
]]
target = merged_data[" Sgpt Alamine Aminotransferase"]

# Handle missing values in features and target
features = features.fillna(features.mean())
target = target.fillna(target.mean())

# Validate non-empty data
if features.shape[0] == 0:
    raise ValueError("Features DataFrame is empty after preprocessing")

# Normalize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Check for NaNs in training/testing sets
print(f"NaNs in X_train: {np.isnan(X_train).sum()}")
print(f"NaNs in y_train: {np.isnan(y_train).sum()}")

# Build the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1)  # Regression output
])

# Learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.0005 * (0.95 ** epoch))

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics=['mae'])

# Train with early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=300, batch_size=64, validation_split=0.2,
                    callbacks=[early_stopping, lr_schedule], verbose=1)

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)


print(f"Test Loss: {test_loss}, Test MAE: {test_mae}, MAE on test set: {mae}")

# Example prediction
new_data = pd.DataFrame(
    [[5, 1.0, 0.5, 200, 30, 7.0, 3.5, 1.0, 5000 / 55, 2.5, 7.4, 50, 5000, 50, 150]],
    columns=[
        "Age of the patient", "Total_Bilirubin", "Direct_Bilirubin",
        " Alkphos Alkaline Phosphotase", "Sgot Aspartate Aminotransferase",
        "Total Protiens", " ALB Albumin", "A/G Ratio Albumin and Globulin Ratio", "ALT/AST Ratio",
        "Partition_Coefficient_LogP", "pKa", "Total_Body_Clearance_CLT",
        "Molecular_Weight", "Protein_Binding_Percent", "IC50_Sodium_Channel_Inhibition"
    ]
)
new_data_scaled = scaler.transform(new_data)
predicted_ALT = model.predict(new_data_scaled)
print(f"Predicted ALT level for new data: {predicted_ALT[0][0]}")

