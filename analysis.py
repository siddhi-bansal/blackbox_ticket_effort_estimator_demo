import ast
import pandas as pd
import matplotlib.pyplot as plt
# Read labels_and_hours from file
labels_and_hours_df = pd.read_csv('labels_and_hours.csv')
import ast

def parse_and_average(hour_entry):
    try:
        values = ast.literal_eval(hour_entry)
        if isinstance(values, list):
            return sum(values) / len(values) if values else 0
        return float(values)
    except Exception:
        return None

labels_and_hours_df['Hours'] = labels_and_hours_df['Hours'].apply(parse_and_average)

average_hours = labels_and_hours_df.groupby('Label')['Hours'].mean().to_dict()
plt.bar(average_hours.keys(), average_hours.values())
plt.xlabel('Labels')
plt.ylabel('Average Hours')
plt.title('Average Hours per Label')
plt.xticks(rotation=45, ha='right')
plt.gcf().set_size_inches(10, 6)
plt.show()
