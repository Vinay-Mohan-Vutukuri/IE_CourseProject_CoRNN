import matplotlib.pyplot as plt

def read_accuracy_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [float(line.strip().split()[-1]) for line in lines]

# Replace 'file1.txt' and 'file2.txt' with the actual paths to your files
file1_accuracies = read_accuracy_file('hand_CoRNN_log.txt')
file2_accuracies = read_accuracy_file('hand_lstm_log.txt')

epochs = list(range(1, len(file1_accuracies) + 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, file1_accuracies, label='CoRNN', marker='o', linestyle='-', color='b')
plt.plot(epochs, file2_accuracies, label='LSTM', marker='s', linestyle='--', color='r')
plt.title('Evaluation Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Evaluation Accuracy')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("hand_plot.png")