import torch
import torch.nn as nn
import numpy as np
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import csv

# Small-SNN(500 Neurons) - LIF Model ("takes the sum of weighted inputs, much like the artificial neuron. But rather than passing it directly to an activation function, it will integrate the input over time with a leakage, much like an RC circuit. If the integrated value exceeds a threshold, then the LIF neuron will emit a voltage spike."" - snnTorch Docs 2.1")
# Lapicque's Model

beta = 0.5  # leak factor
R = 1       # resistance
C = 1.44    # capacitance
batch_size = 500
tau = R * C
num_inputs = 6 # temp, audio, humidity, co2, lat, long
num_outputs = 2 # fire detected / not detected
num_hidden1 = 500
num_hidden2 = 500

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden1)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())
        
        self.fc_out = nn.Linear(num_hidden2, num_outputs)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, mem1=None, mem2=None, mem3=None):
        
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif_out.init_leaky()
            
        spk_rec = []
        mem_rec = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc_out(spk2)
            spk3, mem3 = self.lif_out(cur3, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        return torch.stack(spk_rec), torch.stack(mem_rec)
    
def gen_dry_data():
    # Final clean dataset list
    all_data = []
    fire_count = 0
    nofire_count = 0
    
    for i in range(100000):

        fire = bool(random.randint(0, 1))  # Random class generator
        row = []

        if fire:
            fire_count += 1

            # Temperature (°C)
            temp = round(random.uniform(35, 90), 2)

            # Audio (normalized loudness)
            audio = round(random.uniform(0.40, 1.00), 3)
            
            # Humidity (%)
            humid = round(random.uniform(5, 35), 2)
            
            # CO2 (ppm)
            co2 = round(random.uniform(1200, 5000), 2)
            
            # Coordinates (°)
            lat = round(random.uniform(25, 50), 2)
            
            long = round(random.uniform(-125,-110), 2)
            
            label = 1

        else:
            nofire_count += 1
            
            # Temperature (°C)
            temp = round(random.uniform(10, 45), 2)
            
            # Audio (normalized loudness)
            audio = round(random.uniform(0.00, 0.50), 3)
            
            # Humidity (%)
            humid = round(random.uniform(15, 70), 2)
            
            # CO2 (ppm)
            co2 = round(random.uniform(420, 1300), 2)
            
            # Coordinates (°)
            lat = round(random.uniform(35, 70), 2)
            
            long = round(random.uniform(-120,-60), 2)
            
            label = 0

        # Order: Temp, Audio, Humidity, CO2, Fire Label
        row = [temp, audio, humid, co2, lat, long, label]
        all_data.append(row)

    # Write all rows at once
    with open('firetest_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Temp", "Audio", "Humidity", "CO2", "LATITUDE", "LONGITUDE", "Fire"])  # Add header
        writer.writerows(all_data)

    print(f"Written {fire_count + nofire_count} rows to firetest_data.csv")
    print(f"Fire rows: {fire_count}")
    print(f"No-fire rows: {nofire_count}")
