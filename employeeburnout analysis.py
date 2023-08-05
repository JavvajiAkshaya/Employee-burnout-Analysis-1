import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# PLACE THE FILE URL OF WHICH WE UPLOADED TO OUR DRIVE
FILE_URL = "https://drive.google.com/uc?id=1rcr-rqgsr8bxz7IpkbVDQxVhxeVlqwQE"



# Read the CSV file directly from Google Drive
data = pd.read_csv(FILE_URL)

# Separate features (X) and target (y)
X = data.drop(columns=['burnout'])
y = data['burnout']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print("Classification Report:")
print(classification_report(y_test, y_pred))
