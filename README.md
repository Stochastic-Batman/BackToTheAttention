# BackToTheAttention
Attention + Time Series


## Project Structure

This project follows a **code-first** approach. The core logic is implemented in modular Python scripts within the `src/` directory, while Jupyter notebooks are used for demonstration and analysis.

```text
Martltsera/
├── notebooks/
│   ├── data_and_training.ipynb  # these notebooks are functionally equivalent
│   └── inference.ipynb  # to the Python scripts
├── models/  # Stores trained model weights
├── src/
│   ├── DeTention.py  # Defines the Attention + Time Series Architecture
│   ├── train.py  # Contains the training loop and saving logic
│   └── predict.py  # Inference logic
```

## Setup

Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

`python -m venv btta_venv`

and install requirements with:

`pip install -r requirements.txt`

## Usage

1. You can run the training script directly from the terminal:

```bash
# Default hyperparameters: ticker="PINS", period="3y", L=30, train_ratio=0.8, 
# batch_size=32, lr=0.0001, epochs=100, patience=15, d_model=64, n_heads=4, 
# n_layers=2, ff_hidden_size=256, dropout=0.2, use_avg_pool=True
python src/train.py

# Example with custom values
python src/train.py --ticker "PINS" --epochs 10 --batch_size 32 --L 60 --lr 0.001
```

2. After training, predict the next-day price:

```bash
# Uses the latest L days to predict tomorrow's closing price
python src/predict.py

# Example with custom values
python src/predict.py --model_path models/DeTention.pth --ticker "PINS" --L 30
```

or, if you prefer to use notebooks, first run `notebooks/data_and_training.ipynb` and then `notebooks/inference.ipynb`.