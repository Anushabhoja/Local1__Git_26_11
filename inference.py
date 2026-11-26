import pickle
import numpy as np
from pathlib import Path
model_path = Path("model.pkl")
with model_path.open("rb") as f:
	model = pickle.load(f)
X = np.array([[1.0, 2.0]])
pred = model.predict(X)
print(pred)
