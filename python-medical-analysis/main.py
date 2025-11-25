import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("data/medical_records.csv")

print(data.head())

print(data.columns)

print("Summary Statistics:")
print(data.describe(include='all'))

stroke_data = data[data['Disease_Category'] == 'Stroke']
print("Stroke Data Summary Statistics:")
print(stroke_data.describe(include='all'))

disease_stats = data.groupby('Disease_Category').describe(include='all')
print("Disease Category Summary Statistics:")
print(disease_stats)


sns.boxplot(
    x='Disease_Category',
    y='Age',
    data=data
)
plt.title('Age Distribution by Disease Category')
plt.xlabel('Disease Category')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()


sns.violinplot(
    x='Disease_Category',
    y='Age',
    data=data
)
plt.title('Age Distribution by Disease Category (Violin Plot)')
plt.xlabel('Disease Category')
plt.ylabel('Age')
plt.xticks(rotation=45)
plt.show()