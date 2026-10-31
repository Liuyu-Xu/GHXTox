# NTxPred2

A method for predicting the neurotoxic activity of peptides and proteins.

---

## 📌 Introduction
**NTxPred2** is designed to assist researchers in therapeutic peptide and protein development by providing advanced methods for quantifying and classifying neurotoxic peptides and proteins that target the central nervous system.

It employs large language model word embeddings as features for predicting neurotoxic activity. The final model offers **Prediction, Protein-Scanning, and Design** modules, implemented using machine learning and protein language models.

🔗 Visit the web server for more information: [NTxPred2 Web Server](http://webs.iiitd.edu.in/raghava/ntxpred2)

📖 Please cite relevant content for complete details, including the algorithm behind the approach.

---

## 📚 Reference
Rathore AS, Jain S, Choudhury S, Raghava GPS. A large language model for predicting neurotoxic peptides and neurotoxins. Protein Sci. 2025 Aug;34(8):e70200. 
## Zenodo
https://doi.org/10.5281/zenodo.15030415
---
### 🖼️ NTxPred2 Workflow Representation
![NTxPred2 Workflow](https://github.com/raghavagps/NTxPred2/blob/main/NTxPred2_workflow.png)

## 🧪 Quick Start for Reproducibility

Follow these steps to replicate the core results of our paper:

```bash
# 1. Clone the repository
git clone https://github.com/raghavagps/ntxpred2.git
cd ntxpred2

# 2. Set up the environment (conda recommended)
conda env create -f environment.yml
conda activate NTxPred2

# 3. Download pre-trained models
# Visit: https://webs.iiitd.edu.in/raghava/ntxpred2/download.html
# Download the model ZIP and extract it in the root directory

# 4. Run prediction on sample input
python ntxpred2.py -i example.fasta -o output.csv -m 1 -j 1 -wd working_direcotory_path

```

## 🛠️ Installation Options


### 🔹 PIP Installation
To install NTxPred2 via PIP, run:
```bash
pip install ntxpred2
```
To check available options, type:
```bash
ntxpred2 -h
```

### 🔹 Standalone Installation
NTxPred2 is written in **Python 3** and requires the following dependencies:

#### ✅ Required Libraries
```bash
python=3.10.7
pytorch
```
Additional required packages:
```bash
pip install scikit-learn==1.5.2
pip install pandas==1.5.3
pip install numpy==1.25.2
pip install torch==2.1.0
pip install transformers==4.34.0
pip install joblib==1.4.2
pip install onnxruntime==1.15.1
Bio (Biopython): 1.81
tqdm: 4.64.1
torch: 2.6.0
```

### 🔹 Installation using environment.yml
1. Create a new Conda environment:
```bash
conda env create -f environment.yml
```
2. Activate the environment:
```bash
conda activate NTxPred2
```

---

## ⚠️ Important Note
- Due to the large size of the model file, the model directory has been compressed and uploaded.
- Download the **zip file** from [Download Page](https://webs.iiitd.edu.in/raghava/ntxpred2/download.html).
- **Extract the file** before using the code or model.

---

## 🔬 Classification
**NTxPred2** classifies peptides and proteins as **neurotoxic or non-neurotoxic** based on their primary sequence.

🔹 **Model Options**
- **ESM2-t30 (Peptide Model):** For sequences **7-50 amino acids** long.
- **ET (Protein Model):** For sequences **≥ 51 amino acids**.
- **ET (Combined Model):** For sequences of **mixed length**.
- **Default Model:** ESM2-t30 (Peptide Model), selected for best performance and efficiency.

---

## 🚀 Usage

### 🔹 Minimum Usage
```bash
ntxpred2.py -h
```
To run an example:
```bash
ntxpred2.py -i example.fasta
```

### 🔹 Full Usage
```bash
usage: ntxpred2.py [-h]
                   [-i INPUT]
                   [-o OUTPUT]
                   [-t THRESHOLD]
                   [-j {1,2,3,4}]
                   [-m {1,2,3}]
                   [-d {1,2}]
                   [-wd WORKING DIRECTORY]
```
#### Required Arguments
| Argument | Description |
|----------|-------------|
| `-i INPUT` | Input: Peptide or protein sequence (FASTA format or simple format) |
| `-o OUTPUT` | Output file (default: `outfile.csv`) |
| `-t THRESHOLD` | Threshold (0-1, default: `0.5`) |
| `-j {1,2,3,4}` | Job type: 1-Prediction, 2-Protein Scanning, 3-Design, 4-Design all possible mutants |
| `-m {1,2,3}` | Model selection: 1-ESM2-t30 (Peptides), 2-ET (Proteins), 3-ET (Combined) |
| `-wd WORKING` | Working directory for saving results |

---

## 📂 Input & Output Files

### ✅ **Input File Format**
NTxPred2 supports two formats:
1. **FASTA Format:** (Example: `example.fasta`)
2. **Simple Format:** (Example: `example.seq`, each sequence on a new line)

### ✅ **Output File**
- Results are saved in **CSV format**.
- If no output file is specified, results are stored in `outfile.csv`.

---

## 🔍 Jobs & Features

### 🔹 **Job Types**
| Job | Description |
|-----|-------------|
| 1️⃣ **Prediction** | Predicts whether input peptide/protein is neurotoxic or not. |
| 2️⃣ **Protein Scanning** | Identifies neurotoxic regions in a protein sequence. |
| 3️⃣ **Design** | Generates mutant peptides/proteins with a **single amino acid/dipeptide** at a specified position. |
| 4️⃣ **Design All Possible Mutants** | Generates and predicts **all possible mutants**. |

### 🔹 **Additional Options**
| Option | Description |
|--------|-------------|
| `-p POSITION` | Position to insert mutation (1-indexed) |
| `-r RESIDUES` | Mutated residues (single/double letter amino acid codes) |
| `-w {8-20}` | Window length (Protein Scan mode only, default: 12) |
| `-d {1,2}` | Display: 1-Neurotoxic only, 2-All peptides (default) |

---

## 📑 Package Contents

| File | Description |
|------|-------------|
| **INSTALLATION** | Installation instructions |
| **LICENSE** | License information |
| **README.md** | This file |
| **ntxpred2.py** | Python program for classification |
| **example.fasta** | Example file (FASTA format) |

---

## 📦 PIP Installation (Again for Reference)
```bash
pip install ntxpred2
```
Check options:
```bash
ntxpred2 -h
```

---

🚀 **Start predicting neurotoxicity with NTxPred2 today!**

🔗 Visit: [NTxPred2 Web Server](http://webs.iiitd.edu.in/raghava/ntxpred2)

