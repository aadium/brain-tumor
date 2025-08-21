# Brain Tumor Classification

This project implements a neural network from scratch for classifying brain MRI images as either 'healthy' or 'tumor'. The neural network is implemented in `NeuralNet.py` and can be trained and evaluated using the provided Jupyter notebook `classify.ipynb`.

## Project Structure

```
brain-tumor/
├── classify.ipynb         # Jupyter notebook
├── NeuralNet.py           # Neural network class
└── dataset/               # Dataset directory
    ├── h/                 # Subdirectory for 'healthy' images
    └── t/                 # Subdirectory for 'tumor' images
```

## Dataset
I used this dataset: https://www.kaggle.com/datasets/preetviradiya/brian-tumor-dataset

## How to Run
1. **Install dependencies**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn opencv-python
   ```
2. **Open `classify.ipynb` in Jupyter or VS Code**.
3. **Run all cells** to:
   - Load and preprocess the dataset
   - Visualize sample images
   - Train a single-layer and double-layer neural network
   - Evaluate and visualize predictions

## Key Files
- `classify.ipynb`: Main workflow for data loading, preprocessing, training, and evaluation.
- `NeuralNet.py`: Contains the neural network model and training logic.

## Customization
- Change `pixels` in the notebook to adjust image size.
- Update `LABEL_MAP` if you add more classes.
- Adjust neural network hyperparameters (e.g., `D1`, `D2`, `num_epochs`, `lr`) as needed.

## Notes
- Ensure the dataset is properly organized before running the notebook.
- The code expects grayscale images; color images are automatically converted.
- For best results, use a balanced dataset for each class.
