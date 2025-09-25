import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
data = pd.read_csv("data/malnutrition_sample.csv")

# Indicators to plot
indicators = ['Stunted', 'Underweight', 'Wasted', 'VitaminA_Deficiency', 'Iodine_Deficiency']

# Plot stacked bar chart
plt.figure(figsize=(10,6))
bottom_values = [0]*len(data)  # start at 0 for stacking

for indicator in indicators:
    plt.bar(data['District'], data[indicator], bottom=bottom_values, label=indicator)
    # update bottom for next stack
    bottom_values = [i+j for i,j in zip(bottom_values, data[indicator])]

plt.title('Malnutrition Indicators per District')
plt.xlabel('District')
plt.ylabel('Number of Children')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
