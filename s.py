import pandas as pd
import re

# Read the log file
with open('training_log.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# Prepare data
rows = []
for line in lines:
    # Example: Epoch 1: Gen 5%, Real 95%, Accuracy: 0.1234
    m = re.match(r"Epoch (\d+): Gen (\d+)%?, Real (\d+)%?, Accuracy: ([\d.]+)", line)
    if m:
        epoch, gen, real, acc = m.groups()
        # Fill remaining columns with None or empty string
        row = [int(epoch), int(gen), int(real), float(acc), '', '', '', '']
        rows.append(row)

# Create DataFrame
columns = ['Epoch', 'GenPercent', 'RealPercent', 'Accuracy', 'Col5', 'Col6', 'Col7', 'Col8']
df = pd.DataFrame(rows, columns=columns)

# Save to CSV or print
df.to_csv('training_log_table.csv', index=False)

# Calculate and print standard deviation of Accuracy
std_dev = df['Accuracy'].std()
print(f"Standard deviation of Accuracy: {std_dev}")
