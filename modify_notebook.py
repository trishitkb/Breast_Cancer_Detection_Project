import nbformat
import re

nb_path = 'notebook/breast_cancer_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        # Replace scoring='f1' with scoring='recall'
        if "scoring='f1'" in cell.source:
            cell.source = cell.source.replace("scoring='f1'", "scoring='recall'")
        
        # Replace 'models/best_model.pkl' with 'models/best_model_v1.0.pkl'
        if "'../models/best_model.pkl'" in cell.source:
            cell.source = cell.source.replace("'../models/best_model.pkl'", "'../models/best_model_v1.0.pkl'")
            
with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook updated successfully.")
