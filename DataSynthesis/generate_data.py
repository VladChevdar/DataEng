# Vlad Chevdar | DataEng S25 - Data Synthesis Lab Assignment
import pandas as pd
import numpy as np
from faker import Faker
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

NUM_EMPLOYEES = 10_000
employees = []
faker = Faker('en_US')
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# Load files
departments_df = pd.read_csv('departments_roles.csv')
roles_df = pd.read_csv('roles_and_salaries.csv')

departments_df['% of employees'] = departments_df['% of employees'].str.rstrip('%').astype(float) / 100.0

# Expand departments list based on distribution
department_choices = np.random.choice(
    departments_df['Department'],
    size=NUM_EMPLOYEES,
    p=departments_df['% of employees'].values
)

dept_roles_map = roles_df.groupby('Department')['Role'].apply(list).to_dict()
roles_df['Lower'] = roles_df['Lower'].replace(r'[\$,]', '', regex=True).astype(float)
roles_df['Upper'] = roles_df['Upper'].replace(r'[\$,]', '', regex=True).astype(float)
salary_bounds = roles_df.set_index('Role')[['Lower', 'Upper']].to_dict(orient='index')

for i in range(NUM_EMPLOYEES):
    employee_id = 100000000 + i
    department = department_choices[i]
    role = random.choice(dept_roles_map[department])
    salary_range = salary_bounds[role]
    salary = int(np.random.uniform(salary_range['Lower'], salary_range['Upper']))

    birthdate = faker.date_of_birth(minimum_age=20, maximum_age=65)
    min_hire_date = pd.to_datetime(birthdate) + pd.DateOffset(years=20)
    max_hire_date = pd.to_datetime("2025-06-01")
    
    if min_hire_date > max_hire_date:
        min_hire_date = max_hire_date - pd.DateOffset(years=1)
    
    hiredate = faker.date_between_dates(date_start=min_hire_date,
                                      date_end=max_hire_date)

    gender = np.random.choice(['female', 'male', 'nonbinary'], p=[0.49, 0.49, 0.02])
    country = np.random.choice(['USA', 'India', 'China', 'Mexico', 'Canada', 'Philippines', 'Taiwan', 'South Korea'],
                               p=[0.6, 0.1, 0.08, 0.06, 0.05, 0.04, 0.04, 0.03])
    
    name = faker.name()
    phone = faker.phone_number()
    ssid = faker.unique.ssn()
    email = f"{name.lower().replace(' ', '.').replace(',', '')}{employee_id % 10000}@example.com"

    employees.append({
        'employeeID': employee_id,
        'CountryOfBirth': country,
        'name': name,
        'phone': phone,
        'email': email,
        'gender': gender,
        'birthdate': birthdate,
        'hiredate': hiredate,
        'department': department,
        'role': role,
        'salary': salary,
        'SSID': ssid
    })

# Create DataFrame
emp_df = pd.DataFrame(employees)

# Save emp_df to CSV
emp_df.to_csv('employee_data.csv', index=False)

# Calculate age for each employee
current_date = datetime(2024, 6, 1)  # Using a fixed current date
emp_df['age'] = ((current_date - pd.to_datetime(emp_df['birthdate'])).dt.days / 365.25).astype(int)

# Create sampling weights based on age
weights = np.ones(len(emp_df))
weights[(emp_df['age'] >= 40) & (emp_df['age'] < 50)] = 3  # Triple the weight for ages 40-49

# Create the biased sample
smpl_df = emp_df.sample(n=500, weights=weights, random_state=42)

# Create perturbed salary data
prtrb_df = emp_df.copy()
mean_salary = emp_df['salary'].mean()
std_dev = mean_salary * 0.05  # 5% of mean salary as standard deviation
noise = np.random.normal(0, std_dev, size=len(emp_df))
prtrb_df['salary'] = prtrb_df['salary'] + noise
prtrb_df['salary'] = prtrb_df['salary'].round(2)  # Round to 2 decimal places

# Print output
print("\n=== Original Employee Data (emp_df) ===")
print("\n--- emp_df.describe(include='all') ---")
pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
print(emp_df.describe(include='all', percentiles=[.25, .5, .75]))

print("\n--- emp_df.head(10) ---")
print(emp_df.head(10))

print("\n=== Perturbed Data (prtrb_df) ===")
print("\n--- prtrb_df.describe(include='all') ---")
print(prtrb_df.describe(include='all', percentiles=[.25, .5, .75]))

print("\n--- prtrb_df.head(10) ---")
print(prtrb_df.head(10))

print("\n=== Sampled Data (smpl_df) ===")
print("\n--- smpl_df.describe(include='all') ---")
print(smpl_df.describe(include='all', percentiles=[.25, .5, .75]))

print("\n--- smpl_df.head(10) ---")
print(smpl_df.head(10))

total_payroll = emp_df['salary'].sum()
print(f"\n--- Total Yearly Payroll: ${total_payroll:,.2f} ---")

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

plt.style.use('default')

# A. Country of Birth bar chart
plt.figure(figsize=(12, 6))
country_counts = emp_df['CountryOfBirth'].value_counts()
sns.barplot(x=country_counts.index, y=country_counts.values)
plt.title('Employee Count by Country of Birth')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/country_of_birth.png')
plt.close()

# B. Department bar chart
plt.figure(figsize=(12, 6))
dept_counts = emp_df['department'].value_counts()
sns.barplot(x=dept_counts.index, y=dept_counts.values)
plt.title('Employee Count by Department')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/department_counts.png')
plt.close()

# C. Day of week hiring bar chart
plt.figure(figsize=(12, 6))
emp_df['hire_day'] = pd.to_datetime(emp_df['hiredate']).dt.day_name()
day_counts = emp_df['hire_day'].value_counts()
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_counts = day_counts.reindex(days_order)
sns.barplot(x=day_counts.index, y=day_counts.values)
plt.title('Employee Hires by Day of Week')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/hire_day_counts.png')
plt.close()

# D. Salary KDE plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=emp_df['salary'])
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('plots/salary_kde.png')
plt.close()

# E. Birth year line plot
plt.figure(figsize=(12, 6))
emp_df['birth_year'] = pd.to_datetime(emp_df['birthdate']).dt.year
birth_year_counts = emp_df['birth_year'].value_counts().sort_index()
plt.plot(birth_year_counts.index, birth_year_counts.values)
plt.title('Employee Birth Years Distribution')
plt.xlabel('Birth Year')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/birth_year_distribution.png')
plt.close()

# F. Department salary KDE plots
plt.figure(figsize=(15, 8))
sns.kdeplot(data=emp_df, x='salary', hue='department', label='Department')
plt.title('Salary Distribution by Department')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.legend(title='Department', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('plots/department_salary_kde.png')
plt.close()
