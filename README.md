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
│   ├── DeAttention.py  # Defines the Attention + Time Series Architecture
│   ├── train.py  # Contains the training loop and saving logic
│   └── predict.py  # Inference logic
```

## Setup

Check your Python version with `python --version`. If it is not already Python 3.14, set it to 3.14. Then create a virtual environment with:

`python -m venv btta_venv`

and install requirements with:

`pip install -r requirements.txt`