
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Load the Excel file
file_path = "data/loss.xlsx"  # Replace with your actual file path
data = pd.read_excel(file_path)

# Extract data for plotting
training_step = data.iloc[:, 0]  # Assuming the first column contains epoch numbers
model_names = data.columns[1:]  # Assuming subsequent columns contain loss values for each model
loss_data = data.iloc[:, 1:]  # Exclude the first column for losses

# Plot configuration
plt.rcParams['font.family'] = 'Times New Roman'
# plt.figure(figsize=(10, 6))
plt.figure(figsize=(7.5, 4.5))

colors = ["#A51C36", "#4485C7", "#84BA42", "#DBB428"] # "#84BA42", "#DBB428"
markers = ['o', 's', '^', 'D', '*']  # Circle, square, triangle, rhombus, *

for index, model in enumerate(model_names):
    plt.plot(training_step, loss_data[model], label=model, linewidth=2, color=colors[index], marker=markers[index % len(markers)], markersize=4)
    # plt.plot(training_step[::5], loss_data[model][::5], marker=markers[index % len(markers)], linestyle="None",  markersize=7, color=colors[index])


# 设置x轴刻度，使其显示更合理美观，这里以间隔为5显示刻度标签，你可以根据实际情况调整间隔值
plt.xticks(training_step[::2], fontsize=12)

# Add labels, title, and legend
plt.xlabel("Training step", fontsize=14)
plt.ylabel("Loss", fontsize=14)
# plt.title("Training Loss Curves of Different Models", fontsize=16)
plt.legend(title="Models", fontsize=12, loc='best')  # loc='best'让图例自动选择一个相对不遮挡图形的最佳位置
plt.grid(True, linestyle="--", alpha=0.6, which='both')

# Save and display the plot
plt.tight_layout()
output_file = "Fig/loss_curves.png"
plt.savefig(output_file, dpi=300)
# plt.show()

print(f"Plot saved as {output_file}")