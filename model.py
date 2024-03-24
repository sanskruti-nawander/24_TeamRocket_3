#NOT TO BE USED


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Expanded sample data with technical, non-technical, labor-related, and communication jobs
data = {
    'skills': [
        'Python, Java', 'Java, SQL', 'SQL, Python', 'R', 'JavaScript', 'Python, JavaScript',
        'Microsoft Office', 'Excel, Communication', 'Customer Service', 'Marketing, Social Media',
        'Writing, Editing', 'Graphic Design', 'Research, Analysis', 'Administrative Support', 'Data Entry',
        'Presentation Skills', 'Time Management', 'Problem Solving', 'Leadership', 'Teamwork',
        'Organization', 'Multitasking', 'Adaptability', 'Decision Making', 'Creativity'
    ],
    'qualification': [
        'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor',
        'High School Diploma', 'Associate Degree', 'Bachelor', 'Bachelor',
        'Bachelor', 'Associate Degree', 'Master', 'High School Diploma', 'High School Diploma',
        'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor',
        'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor'
    ],
    'job_title': [
        'Software Engineer', 'Data Scientist', 'Data Engineer', 'Data Analyst', 'Web Developer',
        'Full Stack Developer', 'Office Manager', 'Executive Assistant', 'Customer Service Representative', 'Marketing Coordinator',
        'Content Writer', 'Graphic Designer', 'Market Research Analyst', 'Administrative Assistant', 'Data Entry Clerk',
        'Presentation Specialist', 'Time Management Specialist', 'Problem Solving Specialist', 'Leadership Trainer', 'Teamwork Facilitator',
        'Organizational Specialist', 'Multitasking Expert', 'Adaptability Trainer', 'Decision Making Coach', 'Creativity Consultant'
    ],
    'avg_salary': [
        90000, 110000, 95000, 80000, 85000,
        105000, 55000, 48000, 40000, 45000,
        48000, 52000, 60000, 40000, 35000,
        45000, 42000, 40000, 50000, 48000,
        38000, 40000, 45000, 36000, 42000
    ]
}

df = pd.DataFrame(data)

# Splitting data into features and target
X = df[['skills', 'qualification']]
y = df['avg_salary']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = []
categorical_features = ['skills', 'qualification']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Training the model
model.fit(X_train, y_train)

# Predicting on test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

# Take input from the user
user_skills = input("Enter skills (comma-separated): ")
user_qualification = input("Enter qualification: ")

# Create a DataFrame from the user input
new_data = {
    'skills': [user_skills],
    'qualification': [user_qualification]
}

new_df = pd.DataFrame(new_data)

# Make predictions using the trained model
prediction = model.predict(new_df)

# Find the predicted job role based on the maximum predicted salary
predicted_job_index = y_pred.argmax()
predicted_job_role = df.loc[predicted_job_index, 'job_title']

# Display the predicted average salary and job role
print(f'Predicted average salary for the provided skills and qualification: ${prediction[0]:,.2f}')
print(f'Predicted job role: {predicted_job_role}')

