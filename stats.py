import pandas as pd
import re

# Read the log file
with open('training_log.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Prepare data
rows = []
for line in lines:
    m = re.match(r"Epoch (\d+): Gen (\d+)%?, Real (\d+)%?, Accuracy: ([\d.]+)", line)
    if m:
        epoch, gen, real, acc = m.groups()
        # Fill remaining columns with None or empty string
        row = [int(epoch), int(gen), int(real), float(acc)]
        rows.append(row)

# Create DataFrame
columns = ['Epoch', 'GenPercent', 'RealPercent', 'Accuracy']
df = pd.DataFrame(rows, columns=columns)

# Save to CSV or print
df.to_csv('training_log_table.csv', index=False)




'''
# Print number of samples
num_samples = len(df)
print(f"Number of samples: {num_samples}")

# Print number of samples for each GenPercent category
cat_counts = df['GenPercent'].value_counts().sort_index()
print("\nNumber of samples per GenPercent category:")
for gen, count in cat_counts.items():
    print(f"GenPercent {gen}%: {count} samples")
'''

# Calculate and print standard deviation of Accuracy
std_dev = df['Accuracy'].std()
print(f"Standard deviation of Accuracy: {std_dev}")

# Group by GenPercent and RealPercent, then calculate mean Accuracy for each group
grouped = df.groupby(['GenPercent', 'RealPercent'])['Accuracy'].mean().reset_index()

print("\nMean Accuracy for each (GenPercent, RealPercent) group:")
print(grouped)

# Plot GenPercent vs mean Accuracy
import matplotlib.pyplot as plt

# Calculate mean and standard error for each GenPercent
gen_stats = df.groupby('GenPercent')['Accuracy'].agg(['mean', 'count', 'std']).reset_index()
gen_stats['sem'] = gen_stats['std'] / gen_stats['count']**0.5

plt.figure()
plt.errorbar(gen_stats['GenPercent'], gen_stats['mean'], yerr=gen_stats['sem'], fmt='o-', capsize=5)
plt.xlabel('GenPercent')
plt.ylabel('Mean Accuracy')
plt.title('GenPercent vs Mean Accuracy (with Standard Error)')


plt.grid(True)
plt.show()


# Boxplot of Accuracy grouped by GenPercent
plt.figure()
df.boxplot(column='Accuracy', by='GenPercent', grid=True)
plt.xlabel('GenPercent')
plt.ylabel('Accuracy')
plt.title('Boxplot: Accuracy by GenPercent')
plt.suptitle('')  # Remove default pandas boxplot title
plt.grid(True)
plt.show()

# Linear regression: GenPercent vs mean Accuracy
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np

# Use the grouped mean accuracy per GenPercent
X = gen_stats['GenPercent'].values.reshape(-1, 1)
y = gen_stats['mean'].values

# Fit linear regression
lr = LinearRegression()
lr.fit(X, y)
r_squared = lr.score(X, y)

# Get p-value using scipy.stats.linregress
slope, intercept, r_value, p_value, std_err = stats.linregress(gen_stats['GenPercent'], gen_stats['mean'])

print(f"\nLinear regression results:")
print(f"R-squared: {r_squared}")
print(f"p-value: {p_value}")
