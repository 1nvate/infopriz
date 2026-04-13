import kagglehub
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("gyanashish/healthcare-diabetes")
filename = os.listdir(path)[0]
ds = pd.read_csv(os.path.join(path, filename))

print("Средние значения признаков по группам...")
print(ds.groupby('Outcome').mean())

plt.figure(figsize=(16, 8))
correlation_matrix = ds.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Матрица корреляций признаков")
plt.show()

target_corr = correlation_matrix['Outcome'].sort_values(ascending=False)
print("\nКорреляция признаков с Outcome (болезнью)...")
print(target_corr)
