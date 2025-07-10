## Biomarker discovery pipeline

A high-performance biomarker discovery pipeline designed to prioritize **essential genes** using **DepMap_essentiality_score** as a target. This AI-driven framework integrates classical machine learning, deep learning, and AutoML with SHAP-based explainability to identify and evaluate disease-relevant molecular features.

## Purpose

This pipeline was developed for identifying **functionally essential biomarkers** across comorbid pediatric diseases using transcriptomic signatures. The goal is to facilitate downstream experimental validation and drug targeting efforts in line with **DepMap CRISPR-based gene essentiality** data.

---

## 🔬 Key Features

- **Data-driven selection of differentially expressed genes**
- **Feature selection** using Random Forest
- **AutoML training** via [AutoGluon](https://auto.gluon.ai/)
- **Model interpretation** with SHAP (TreeExplainer or KernelExplainer fallback)
- **Top 100 gene candidates** enriched with:
  - [GeneCards links](https://www.genecards.org)
  - [STRING PPI network links](https://string-db.org)
- **Cross-validated ROC and PR curves**
- **Model comparison metrics**: ROC AUC, PR AUC, F1, Recall, Accuracy

---

## 📁 File Structure

```

.
├── ML\_FINAL\_F.csv                      # Input data (DGE matrix + features)
├── FINAL\_AI\_ML\_2121/                  # All generated outputs
│   ├── Selected\_Features\_With\_Label.csv
│   ├── Feature\_Importance\_RF\_Selected.png
│   ├── SHAP\_Summary\_AutoML.png
│   ├── Predictions.csv
│   ├── AutoML\_Model\_Performance.csv
│   ├── CrossVal\_Results\_Expanded.csv
│   ├── CrossVal\_ROC\_Curves.html
│   ├── CrossVal\_PR\_Curves.html
│   ├── Model\_Comparison\_\*.png
│   └── TOP\_100\_CANDIDATE\_GENES.csv
└── natureplusplus\_pipeline.py         # Main pipeline script

````

---

## ⚙️ How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

> Requires Python ≥ 3.8 and AutoGluon ≥ 0.8

### 2. Place Input File

Place the file `ML_FINAL_F.csv` in the root directory of the repo, or modify the `input_file` path in the script.

### 3. Execute the Pipeline

```bash
python natureplusplus_pipeline.py
```

---

## Input Description

**`ML_FINAL_F.csv`** contains:

* Normalized gene expression features
* Metadata columns: `GEO_ID`, `Disease_Name`, `Regulation`, `Gene_Symbol`
* Target column: `DepMap_essentiality_score`
* Output target is `Target` = 1 if score ≤ -0.5, else 0

---

## Machine Learning Models Used

* **Random Forest** (feature selection + evaluation)
* **XGBoost**, **LightGBM**, **CatBoost**
* **Multi-layer Perceptron**
* **AutoML via AutoGluon**

Performance metrics are saved and visualized for all models using 5-fold stratified cross-validation.

---

## Visual Outputs

* Bar plot of **top RF features**
* SHAP summary plot of AutoML-selected features
* Cross-validated **ROC** and **Precision-Recall** curves
* Model performance comparison across five metrics

---

##  Citation & Zenodo

📦 Dataset (ML\_FINAL\_F.csv) is archived at:
🔗 [https://zenodo.org/record/15855304](https://zenodo.org/record/15855304)

Please cite as:

```
Ahammad, F. (2025). AutoML-Guided Discovery of Hub Genes in Cerebral Palsy Reveals Shared Pediatric Disease Mechanisms (V.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15855304

(Zenodo. https://doi.org/10.5281/zenodo.15855304)
```

---

## 🛡️ License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** License.

---

## 🧪 Acknowledgements

This work was developed under support from the College of Health and Life Sciences, Hamad Bin Khalifa University (HBKU), Qatar Foundation. The authors acknowledge use of the [DepMap](https://depmap.org) dataset and [AutoGluon](https://auto.gluon.ai/) libraries.

---

## 💬 Contact

For questions or contributions, please contact:
📧 [Foysal Ahammad](mailto:foah48505@hbku.edu.qa)

