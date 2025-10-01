import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import files
url = "https://raw.githubusercontent.com/j-song-npc/Data622/refs/heads/main/bank-full.csv"
bank_data = pd.read_csv(url, sep=";")

## Data overview 
# Preview data
print(bank_data.head())
print(bank_data.tail())

# View number of observations and features  
print(bank_data.shape) #There are 45,211 data entries and 17 features 

# View data types
print(bank_data.info()) # There is a mix of categorical/numeric features 

# View unique values per variable
print(bank_data.nunique())  # There seem to be a few binary features (default, housing, loan, y)

# Summarize categorical data 
print(bank_data.describe(include='object'))  

# Check duplicated rows
print(f"Duplicate rows: {bank_data.duplicated().sum()}") # There are no duplications

# Check for missing values
print(bank_data.isnull().sum()) # There are no missing values


## Visualization and data review

# Review continuous data 
print(bank_data.describe())

# Visualize continous data
# Most of the data here seems skewed to the right
plt.hist(bank_data['age'], bins=30)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age')
plt.show() 

plt.hist(bank_data['balance'], bins=30)
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.title('Balance')
plt.show()  

plt.hist(bank_data['day'], bins=30)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Day')
plt.show() 

plt.hist(bank_data['duration'], bins=30)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Duration')
plt.show() 

plt.hist(bank_data['campaign'], bins=30)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Campaign')
plt.show()  

plt.hist(bank_data['pdays'], bins=30)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('PDays')
plt.show()

plt.hist(bank_data['previous'], bins=30)
plt.xlabel('Column Name')
plt.ylabel('Frequency')
plt.title('Previous')
plt.show()


# Visualize categorical data 
obj_col = bank_data.select_dtypes(include='object').columns
print(obj_col)

for col in obj_col:
    bank_data[col].value_counts().plot(kind='bar')
    plt.title(col)
    plt.show()

# View binary data 
binary_count = pd.DataFrame({
    col: bank_data[col].value_counts() 
    for col in ['default', 'housing', 'loan', 'y']})
print(binary_count)

# Target (y) value distribution 
print(bank_data['y'].value_counts())
print(bank_data['y'].value_counts(normalize=True) * 100)

# Visualize outliers in numerical data 
sns.boxplot(x=bank_data['age'])
sns.boxplot(x=bank_data['balance']) # There appears to be significant outliers in 'balance'
sns.boxplot(x=bank_data['day'])
sns.boxplot(x=bank_data['duration']) # There appears to be significant outliers in 'duration'
sns.boxplot(x=bank_data['campaign']) # There appears to be significant outliers in 'campaign'
sns.boxplot(x=bank_data['pdays'])
sns.boxplot(x=bank_data['previous']) # There appears to be significant outliers in 'previous'


# Check correlation of numeric features
correl = bank_data.corr(numeric_only=True)
sns.heatmap(correl, annot=False, cmap="coolwarm")
plt.title("Correlation")
plt.show() # There apepars to be a moderate correlation between pdays and previous 

