import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Make sure to install the Pillow library for image handling

from torch import nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import math
from torch.nn import init
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torchvision
import torchvision.transforms as transforms

from torch import nn, optim
import torch
import utils
import network
import argparse
import torch.nn.utils
from pathlib import Path

import pandas as pd

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h =   nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self,x,hy,hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy),1)))
                                   - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz

        return hy, hz

class coRNN(nn.Module):
    def __init__(self, n_inp, n_hid, n_out, dt, gamma, epsilon):
        super(coRNN, self).__init__()
        self.n_hid = n_hid
        self.cell = coRNNCell(n_inp,n_hid,dt,gamma,epsilon)
        self.readout = nn.Linear(n_hid, n_out)

    def forward(self, x):
        ## initialize hidden states
        hy = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        hz = Variable(torch.zeros(x.size(1),self.n_hid)).to(device)
        # print(hy.shape)
        for t in range(x.size(0)):
            hy, hz = self.cell(x[t],hy,hz)
        output = self.readout(hy)

        return output
class InputGUI:
    def __init__(self, master):
        self.master = master
        master.title("Input GUI")

        # Option menu for choosing between MNIST and hand_prediction
        self.options = ["MNIST", "hand_prediction"]
        self.var = tk.StringVar(master)
        self.var.set(self.options[0])  # Set the default value
        self.option_menu = tk.OptionMenu(master, self.var, *self.options)
        self.option_menu.pack()

        self.label_file = tk.Label(master, text="Select input file:")
        self.label_file.pack()

        self.entry_file = tk.Entry(master, state='disabled')  # Entry widget to display selected file path
        self.entry_file.pack()

        self.browse_button = tk.Button(master, text="Browse", command=self.browse_file)
        self.browse_button.pack()

        self.submit_button = tk.Button(master, text="Submit", command=self.store_inputs)
        self.submit_button.pack()

        # Canvas to display images
        self.canvas = tk.Canvas(master, width=300, height=300)
        self.canvas.pack()

        # Label to display the predicted class
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        self.entry_file.config(state='normal')
        self.entry_file.delete(0, tk.END)
        self.entry_file.insert(0, file_path)
        self.entry_file.config(state='disabled')

        # Display the selected image on the canvas
        image = Image.open(file_path).resize((300, 300))
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def store_inputs(self):
        selected_option = self.var.get()
        input_file_path = self.entry_file.get()

        if selected_option == "MNIST":
            # Load MNIST pretrained model and test input_file_path
            model = coRNN(1, 256, 10, 0.042, 2.7, 4.7)
            model.load_state_dict(torch.load('pminst_model_checkpoint.pth', map_location=torch.device('cpu')))
            model.eval()

            # Load and preprocess the input image
            image = Image.open(input_file_path).convert('L')  # Convert to grayscale
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
            ])
            image = transform(image)
            image = Variable(image.unsqueeze(0))  # Add batch dimension
            image = image.reshape(image.shape[2] * image.shape[3], image.shape[0], image.shape[1])

            # Make predictions
            with torch.no_grad():
                output = model(image)

            # Get the predicted class
            _, predicted_class = torch.max(output, 1)

            # Display the predicted class on the interface
            result_text = f"Predicted class: {predicted_class.item()}"
            print(result_text)
            self.result_label.config(text=result_text)

        elif selected_option == "hand_prediction":
            # Load hand_prediction pretrained model and test input_file_path
            hand_prediction_model = coRNN(1, 128, 4, 6e-2, 66, 15)
            hand_prediction_model.load_state_dict(torch.load('best_model_checkpoint.pth', map_location=torch.device('cpu')))
            hand_prediction_model.eval()

            df = pd.read_csv(input_file_path)
            input_data = torch.tensor(df.values, dtype=torch.float32)
            input_data = Variable(input_data.unsqueeze(0))  # Add batch dimension

            input_data_tensor = torch.tensor(input_data, dtype=torch.float32)
            input_data_tensor = input_data_tensor.to(device)
            input_data_tensor = input_data_tensor.reshape(input_data_tensor.shape[2], input_data_tensor.shape[1], 1)

            # Make predictions
            with torch.no_grad():
                output = hand_prediction_model(input_data_tensor)

            Y_pred_classes = torch.argmax(torch.sigmoid(output), axis=1)

            # Display the predicted class on the interface
            result_text = f"Predicted class: {Y_pred_classes.item()}"
            self.result_label.config(text=result_text)

# Storing selected option and input file path
root = tk.Tk()
input_gui = InputGUI(root)
root.mainloop()