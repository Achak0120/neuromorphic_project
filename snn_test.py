from pathlib import Path
import torch
import pandas as pd
from forest_fire_snn import SNN

BASE_DIR = Path(__file__).resolve().parent

net = SNN()
model_path = BASE_DIR / "trained_snn_model.pth"
net.load_state_dict(torch.load(model_path, map_location="cpu"))
net.eval()
print("Trained model loaded successfully.")

csv_path = BASE_DIR / "firedata_validation.csv"
df = pd.read_csv(csv_path)

# normalize once
df["Temp"] = df["Temp"] / 100
df["Humidity"] = df["Humidity"] / 100
df["CO2"] = df["CO2"] / 5000

num_steps = 25

preds, confs, score0, score1 = [], [], [], []

with torch.no_grad():
    for _, row in df.iterrows():
        x0 = torch.tensor([[row["Temp"], row["Audio"], row["Humidity"], row["CO2"]]],
                          dtype=torch.float32)
        x = x0.unsqueeze(0).repeat(num_steps, 1, 1)

        spk_rec, mem_rec = net(x)
        spike_counts = spk_rec.sum(dim=0)

        s0 = float(spike_counts[0, 0].item())
        s1 = float(spike_counts[0, 1].item())
        pred = int(torch.argmax(spike_counts, dim=1).item())
        conf = float(spike_counts[0, pred].item() / (spike_counts[0].sum().item() + 1e-9))

        score0.append(s0); score1.append(s1)
        preds.append(pred); confs.append(conf)

df["Spikes_Class0"] = score0
df["Spikes_Class1"] = score1
df["Pred_Fire"] = preds
df["Pred_Conf"] = confs

out_path = BASE_DIR / "firedata_validation_with_preds.csv"
df.to_csv(out_path, index=False)
print("Saved:", out_path)
print(df)