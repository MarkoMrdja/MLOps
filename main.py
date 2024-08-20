from models.hyperoptimize import run_optimization
from data.load_data import load_data

train_loader, val_loader = load_data()

model = run_optimization(train_loader, val_loader)

print(model)