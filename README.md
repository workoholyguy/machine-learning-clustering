# CSC 6850 — Machine Learning  
## Individual Project — Omar Madjitov

This repository collects the end-to-end work for the CSC 6850 individual project, with an emphasis on building classical machine-learning pipelines from first principles. Every algorithm is implemented in **pure NumPy**, so the focus is on algorithmic intuition, numerical stability, and reproducible experimentation rather than on off-the-shelf libraries. The project bridges exploratory data analysis, robust handling of missing values, and performant classification models, mirroring the typical responsibilities of a machine learning engineer, data scientist, and data analyst.

---

## Project Highlights  
- **Analytical Rigor:** Each dataset is inspected for class balance, feature scales, and missingness, allowing for data-driven decisions in subsequent modeling steps.  
- **Data Engineering:** Custom routines detect missing entries and isolate the largest complete submatrix before imputing absent values. This demonstrates an ability to clean messy inputs without relying on third-party imputation utilities.  
- **Modeling Craftsmanship:** Implements and tunes both K-Nearest Neighbors classifiers and neural-network models (softmax logistic regression and a one-hidden-layer MLP) using gradient descent with L2 regularization, showcasing a spectrum of modeling tools from lazy learning to deep learning basics.  
- **Reproducible Workflow:** Notebook structure documents every experiment, from preprocessing through cross-validated hyperparameter selection to final prediction export, which aligns with production-ready reporting expectations.

## Technical Skills Demonstrated  
- **Data Analysis & Feature Engineering:** Feature-standardization, stratified splitting, cross-tabulation of classes, and dimensionality inspection are all performed manually in `01_data_overview.ipynb`.  
- **Missing Data Strategy:** `02_missing_value.ipynb` covers missing-value detection, row/column filtering, RMSE-driven K selection, and complete imputation pipelines for the provided MissingData1/2 sources.  
- **Model Building & Evaluation:** 
  - Implemented from-scratch KNN with manual distance calculations, per-fold validation, and final predictions captured in `results/MadjitovClassification3.txt` and `results/MadjitovClassification4.txt`.  
  - Automated gradient-descent training for multiclass logistic regression and a ReLU-activated MLP in `05_classification_nn.ipynb`, with early-stopping inspired logic and L2 weight decay to control overfitting.  
- **Result Delivery:** Final label files are stored in `results/` to mimic production inference outputs, while the notebooks narrate the rationale behind every transformation and score.

## Repository Layout  
- `notebooks/01_data_overview.ipynb` — Exploratory data analysis, class distributions, feature scales.  
- `notebooks/02_missing_value.ipynb` — Missing-data detection, dynamic K-selection, and imputation for MissingData1.txt / MissingData2.txt.  
- `notebooks/03_classification_knn.ipynb` — Custom KNN classifier, stratified splits, k-fold validation, and outputs for Datasets 3 and 4.  
- `notebooks/05_classification_nn.ipynb` — Multiclass softmax and MLP implementations, trained via gradient descent/regularization for Datasets 1 and 2.  
- `results/` — Persisted prediction files from both KNN and neural-network pipelines.  
- `StudentProjectData/` — Raw data files referenced by the notebooks.  
- `requirements.txt` — Minimal dependencies required to run the notebooks (NumPy, Pandas, Matplotlib, etc.).

## Workflow Overview
1. **Exploration (`01_data_overview.ipynb`):** Characterizes each dataset’s scale, sparsity, and distribution, which in turn informs preprocessing choices like mean-centering or standard scaling.  
2. **Cleaning (`02_missing_value.ipynb`):** Uses RMSE-based heuristics to pick an optimal K for KNN imputation, keeping track of error curves and ensuring data completeness prior to model training.  
3. **Modeling (`03_classification_knn.ipynb` & `05_classification_nn.ipynb`):** KNN pipelines emphasize distance-based reasoning with stratified validation folds, while neural implementations rely on matrix-vector gradient computations, ReLU activations, and analytic softmax loss to produce class probabilities.  
4. **Evaluation & Export:** Cross-validated K selection and manual inspection of loss surfaces ensure the final choices are defensible; predictions are captured as plain text files for easy sharing or scoring.

## Running the Project
1. **Install dependencies**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```  
2. **Launch notebooks** – each notebook can be opened via Jupyter or VSCode and executed sequentially to reproduce the analysis.  
3. **Data paths** – the notebooks reference files in `StudentProjectData/`, so ensure this directory remains intact.  
4. **Result artifacts** – once completed, check the `results/` directory for final predictions that mirror submission-ready output files.

## Outcome & Impact  
- Demonstrated capability to translate problem statements into reproducible experiments without machine-learning libraries, proving a strong command of linear algebra, optimization, and statistical reasoning.  
- Documented every decision within the notebooks, reinforcing transparency and interpretability for stakeholders reviewing the work.  
- Generated final predictions for all provided datasets, enabling downstream analysis or leaderboard submission using the exported `.txt` files.

## Next Steps  
1. Extend the neural workflows with learning-rate schedules or momentum to mimic more advanced training regimes.  
2. Benchmark the NumPy implementations against scikit-learn counterparts to highlight performance trade-offs.  
3. Package the pipelines for reuse or integrate them into a lightweight CLI for automated scoring.

## Detailed Report  
Read the accompanying project report for a narrative walk-through and analysis artifacts: https://docs.google.com/document/d/1UdppvCVAW6rbBQNalArVim3R9OWPLqfuZF5CJCkNacg/edit?usp=sharing
