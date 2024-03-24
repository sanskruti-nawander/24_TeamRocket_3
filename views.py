from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Define the model and DataFrame
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

X = df[['skills', 'qualification']]
y = df['avg_salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = []
categorical_features = ['skills', 'qualification']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

model.fit(X_train, y_train)

# Define the predict_job view function
def predict_job(request):
    if request.method == 'POST':
        skills = request.POST.get('skills')
        qualification = request.POST.get('qualification')

        # Make predictions using the trained model
        prediction = model.predict(pd.DataFrame({'skills': [skills], 'qualification': [qualification]}))

        # Find the predicted job role based on the maximum predicted salary
        predicted_job_index = prediction.argmax()
        predicted_job_role = df.loc[predicted_job_index, 'job_title']

        context = {
            'prediction': f'Predicted average salary: ${prediction[0]:,.2f}',
            'predicted_job_role': f'Predicted job role: {predicted_job_role}'
        }
        return render(request, 'result.html', context)

    return render(request, 'index.html')


