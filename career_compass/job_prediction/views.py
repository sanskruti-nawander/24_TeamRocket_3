from django.shortcuts import render, redirect
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from .forms import UserLoginForm, UserRegistrationForm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


# Define the model and DataFrame
data = {
    'skills': [
        'Python, Java', 'Java, SQL', 'SQL, Python', 'R', 'JavaScript', 'Python, JavaScript',
        'Microsoft Office', 'Excel, Communication', 'Customer Service', 'Marketing, Social Media',
        'Writing, Editing', 'Graphic Design', 'Research, Analysis', 'Administrative Support', 'Data Entry',
        'Presentation Skills', 'Time Management', 'Problem Solving', 'Leadership', 'Teamwork',
        'Organization', 'Multitasking', 'Adaptability', 'Decision Making', 'Creativity',
        'Project Management', 'Sales', 'Networking', 'Database Management', 'Customer Relationship Management',
        'Public Speaking', 'Business Development', 'Financial Analysis'
    ],
    'qualification': [
        'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor',
        'High School Diploma', 'Associate Degree', 'Bachelor', 'Bachelor',
        'Bachelor', 'Associate Degree', 'Master', 'High School Diploma', 'High School Diploma',
        'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor',
        'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor', 'Bachelor',
        'Master', 'Bachelor', 'Master', 'Bachelor', 'Master',
        'Bachelor', 'Master', 'Bachelor'
    ],
    'job_title': [
        'Software Engineer', 'Data Scientist', 'Data Engineer', 'Data Analyst', 'Web Developer',
        'Full Stack Developer', 'Office Manager', 'Executive Assistant', 'Customer Service Representative', 'Marketing Coordinator',
        'Content Writer', 'Graphic Designer', 'Market Research Analyst', 'Administrative Assistant', 'Data Entry Clerk',
        'Presentation Specialist', 'Time Management Specialist', 'Problem Solving Specialist', 'Leadership Trainer', 'Teamwork Facilitator',
        'Organizational Specialist', 'Multitasking Expert', 'Adaptability Trainer', 'Decision Making Coach', 'Creativity Consultant',
        'Project Manager', 'Sales Representative', 'Network Administrator', 'Database Administrator', 'CRM Specialist',
        'Public Relations Manager', 'Business Analyst', 'Financial Manager'
    ],
    'avg_salary': [
        90000, 110000, 95000, 80000, 85000,
        105000, 55000, 48000, 40000, 45000,
        48000, 52000, 60000, 40000, 35000,
        45000, 42000, 40000, 50000, 48000,
        38000, 40000, 45000, 36000, 42000,
        100000, 75000, 85000, 90000, 80000,
        95000, 85000, 110000
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


def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('predict_job')  # Redirect to job_prediction page after successful login
        else:
            # Handle invalid login
            return render(request, 'login.html', {'error': 'Invalid credentials'})

    return render(request, 'login.html')


def user_register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            confirm_password = form.cleaned_data['confirm_password']
            if password == confirm_password:
                user = User.objects.create_user(username=username, password=password)
                login(request, user)
                return redirect('main_project_view')  # Replace 'main_project_view' with the actual view name
    else:
        form = UserRegistrationForm()
    
    return render(request, 'register.html', {'form': form})


def welcome_page(request):
    return render(request, 'welcome_page.html')


def job_trends(request):
    # Sample data (replace with your actual data)
    years = np.array([2020, 2021, 2022, 2023, 2024])
    salaries = np.array([50000, 55000, 60000, 65000, 70000])

    # Create a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(years, salaries, marker='o', color='b')

    # Add labels and title
    plt.title('Predicted Job Salaries Over Years')
    plt.xlabel('Year')
    plt.ylabel('Average Salary ($)')
    plt.grid(True)

    # Save the graph as a PNG image
    plt.savefig('static/job_trends.png')

    # Display the graph
    plt.show()

