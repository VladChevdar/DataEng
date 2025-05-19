import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from us_state_abbrev import abbrev_to_us_state

dashline = "----------------------------------"

# Read
cases_df = pd.read_csv('covid_confirmed_usafacts.csv')
deaths_df = pd.read_csv('covid_deaths_usafacts.csv')
census_df = pd.read_csv('acs2017_county_data.csv')

# Trim
cases_df = cases_df[['County Name', 'State', '2023-07-23']]
deaths_df = deaths_df[['County Name', 'State', '2023-07-23']]
census_df = census_df[['County', 'State', 'TotalPop', 'IncomePerCap', 'Poverty', 'Unemployment']]

# Display
print(dashline)
print("cases_df columns:", cases_df.columns.tolist())
print("deaths_df columns:", deaths_df.columns.tolist())
print("census_df columns:", census_df.columns.tolist())
print(dashline)

# Integration Challenge #1
cases_df['County Name'] = cases_df['County Name'].str.strip()
deaths_df['County Name'] = deaths_df['County Name'].str.strip()
washington_cases = cases_df[cases_df['County Name'] == 'Washington County']
washington_deaths = deaths_df[deaths_df['County Name'] == 'Washington County']

print("Washington County count in cases_df:", len(washington_cases))
print("Washington County count in deaths_df:", len(washington_deaths))
print(dashline)

# Integration Challenge #2
cases_df = cases_df[cases_df['County Name'] != 'Statewide Unallocated']
deaths_df = deaths_df[deaths_df['County Name'] != 'Statewide Unallocated']

print("Remaining rows in cases_df:", len(cases_df))
print("Remaining rows in deaths_df:", len(deaths_df))
print(dashline)

# Integration Challenge #3 
cases_df['State'] = cases_df['State'].map(abbrev_to_us_state)
deaths_df['State'] = deaths_df['State'].map(abbrev_to_us_state)

print(cases_df.head())
print(dashline)

# Integration Challenge #4 
cases_df['key'] = cases_df['County Name'] + ', ' + cases_df['State']
deaths_df['key'] = deaths_df['County Name'] + ', ' + deaths_df['State']
census_df['key'] = census_df['County'] + ', ' + census_df['State']

cases_df.set_index('key', inplace=True)
deaths_df.set_index('key', inplace=True)
census_df.set_index('key', inplace=True)

print(census_df.head())
print(dashline)

# Integration Challenge #5 
cases_df.rename(columns={'2023-07-23': 'Cases'}, inplace=True)
deaths_df.rename(columns={'2023-07-23': 'Deaths'}, inplace=True)

print("cases_df columns:", cases_df.columns.values.tolist())
print("deaths_df columns:", deaths_df.columns.values.tolist())
print(dashline)

# Do the Integration
join_df = cases_df.join(deaths_df[['Deaths']])
join_df = join_df.join(census_df[['TotalPop', 'IncomePerCap', 'Poverty', 'Unemployment']])

join_df['CasesPerCap'] = join_df['Cases'] / join_df['TotalPop']
join_df['DeathsPerCap'] = join_df['Deaths'] / join_df['TotalPop']

print("Number of rows in join_df:", len(join_df))
print(dashline)

# Correlation matrix
correlation_matrix = join_df.corr(numeric_only=True)
print(correlation_matrix)

# Visualize
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

plt.title('Correlation Matrix Heatmap')
plt.tight_layout()  # Ensure it fits well
plt.show()
