
<p align="center">
  <img src="https://costalab.org/wp-content/uploads/2020/12/cropped-logo-1.png" width="180"/>
</p>

<h1 align="center">
Predicting Cellular Neighborhoods in Spatial Transcriptomics via Morphometrics and Graph Learning
</h1>

<p align="center">
<b>Ivan · Yaroslav · Simay</b>
</p>

<p align="center">
Spatial Transcriptomics • Morphometrics • Graph Learning • Link Prediction
</p>

---

---

# 📚 Table of Contents

- [Abstract](#-abstract)
- [Key Idea](#-key-idea)
- [Pipeline Overview](#-pipeline-overview)
- [Models](#-models)
- [Datasets](#-datasets)
- [Research Questions](#-research-questions)
- [Evaluation Metrics](#-evaluation-metrics)
- [Repository Structure](#-repository-structure)
- [Current Status](#-current-status)
- [Timeline](#-timeline)
- [Contributions](#-key-contributions-expected)
- [Future Work](#-future-work)
- [Paper](#-paper)
- [Citations](#-citations)
- [External References](#-external-references)
- [Contact](#-contact)
- [Acknowledgements](#-acknowledgements)

---

---

# 📖 Abstract

Understanding cellular organization in tissue is a fundamental problem in computational biology.  
We investigate whether **cellular neighborhood relationships can be predicted from gene expression and morphometric features** using classical and graph-based learning methods.

We formulate this as a **graph link prediction problem**, comparing:

- kNN baselines 🔍  
- weighted kNN ⚖️  
- morphology-enhanced models 🧬  
- graph neural networks (GCN, GAT) 🧠  

---

--- 
# 💡 Key Idea

> Can we reconstruct cellular neighborhoods using only molecular + morphological signatures?

---

--- 
# 🔬 Pipeline Overview

```
Raw Spatial Transcriptomics Data
            ↓
Preprocessing & QC 
            ↓
Feature Extraction 
            ↓
Graph Construction 
            ↓
Link Prediction Models 
            ↓
Evaluation & Interpretation 
```

---

---
# 🧪 Models

## Baselines
- k-Nearest Neighbors (kNN)
- Weighted kNN 

## Morphology Models
- Morphology-only kNN
- Combined feature kNN

## Graph Models
- Graph Convolutional Network (GCN)
- Graph Attention Network (GAT)

---

--- 
# 🗂️ Datasets

| Dataset | Cells | 10x version | Type | SHA256 |
|---|---:|---:|---|---|
| [tiny_human_kidney_protein_v4](https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Protein_Human_Kidney_tiny/Xenium_V1_Protein_Human_Kidney_tiny_outs.zip) | 358 | 4.0.0 | tiny; protein | `abd7e8f7fd04…` |
| [tiny_human_ovary_multimodal_v4](https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_MultiCellSeg_Human_Ovary_tiny/Xenium_V1_MultiCellSeg_Human_Ovary_tiny_outs.zip) | 623 | 4.0.0 | tiny; multimodal | `9b154a3c0836…` |
| [tiny_human_ovary_nucexp_v4](https://cf.10xgenomics.com/samples/xenium/4.0.0/Xenium_V1_Human_Ovary_tiny/Xenium_V1_Human_Ovary_tiny_outs.zip) | 632 | 4.0.0 | tiny; nucleus expansion | `72b9a6d73ec4…` |
| [tiny_mouse_ileum_multimodal_v3](https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny/Xenium_Prime_MultiCellSeg_Mouse_Ileum_tiny_outs.zip) | 36 | 3.0.0 | tiny; multimodal | `c4d028676e8b…` |
| [tiny_mouse_ileum_nucexp_v3](https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Mouse_Ileum_tiny/Xenium_Prime_Mouse_Ileum_tiny_outs.zip) | 23 | 3.0.0 | tiny; nucleus expansion | `72dc2353825f…` |
| [breast_2fov_v2](https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Breast_2fov/Xenium_V1_human_Breast_2fov_outs.zip) | 7,275 | 2.0.0 | 2 FOV; Xenium | `cc1e987b06aa…` |
| [lung_2fov_v2](https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_human_Lung_2fov/Xenium_V1_human_Lung_2fov_outs.zip) | 11,898 | 2.0.0 | 2 FOV; Xenium | `acc353069871…` |
| [prostate_prime_5k_v3](https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Prostate_FFPE/Xenium_Prime_Human_Prostate_FFPE_outs.zip) | 193,000 | 3.0.0 | large; Xenium Prime | `5eeee73d9f0c…` |
| [ovarian_prime_5k_v3](https://s3-us-west-2.amazonaws.com/10x.files/samples/xenium/3.0.0/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun/Xenium_Prime_Ovarian_Cancer_FFPE_XRrun_outs.zip) | 407,124 | 3.0.0 | large; Xenium Prime | `1ba372a8198c…` |
| [lymph_node_prime_5k_v3](https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE/Xenium_Prime_Human_Lymph_Node_Reactive_FFPE_outs.zip) | 708,983 | 3.0.0 | large; Xenium Prime | `1931b2e45dbf…` |

---

---
# ❓ Research Questions

- Does morphology improve neighbor prediction? 
- Can gene expression alone reconstruct spatial structure? 
- Do attention weights reflect biology? 
- Which modality is most informative? 

---

---
# 📏 Evaluation Metrics

- Accuracy 
- Precision / Recall  
- F1-score  
- ROC-AUC  
- Runtime  

---

---
# 📁 Repository Structure

Proposed structure, not up to date.

```text
├── docs/              📖 Documentation
├── data/              🗂️ Raw + processed datasets
├── src/               ⚙️ Core pipeline
│   ├── models/
│   ├── features/
│   ├── preprocessing/
│   └── evaluation/
├── experiments/       🧪 Reproducible runs
├── results/           📊 Figures + tables
├── notebooks/         📓 Exploration
├── paper/             📄 Final paper
└── presentation/      🎤 Slides
```

---

---
# 📊 Current Status

## ✅ Completed
- Project definition
- Dataset exploration
- Literature review setup

## 🟡 In Progress
- Baseline kNN implementation
- HPC setup
- Morphology feature extraction

## 🔵 Planned
- GCN / GAT experiments
- Ablation studies
- Biological interpretation

---

---
# ⏳ Timeline

| Phase | Date | Goal |
|------|------|------|
| Setup | May 2026 | Data + baselines |
| Development | May–June 2026 | Models + experiments |
| Finalization | June 2026 | Analysis + paper |
| Presentation | July 6, 2026 | Defense |

---

---
# 🏆 Key Contributions (Expected)

- Comparative analysis of transcriptomics vs morphology 
- Graph-based framework for spatial biology 
- Benchmark of classical vs neural models 
- Biological interpretation of spatial structure 

---

---
# 🚀 Future Work

- 

---

---
# 📄 Paper

📌 **Working Paper (Draft / Coming Soon)**  
> 

---

---
# 📚 Citations

## Spatial Transcriptomics & Morphometrics

- Chelebian, E., Avenel, C., & Wählby, C. (2025). *Combining spatial transcriptomics with tissue morphology*. Nature Communications, 16(1), 4452.  
  https://doi.org/10.1038/s41467-025-58989-8

- Hallou, A., He, R., Simons, B. D., & Dumitrascu, B. (2025). *A computational pipeline for spatial mechano-transcriptomics*. Nature Methods, 22(4), 737–750.  
  https://doi.org/10.1038/s41592-025-02618-1

---

## Graph Neural Networks & Link Prediction

- Chen, G., & Liu, Z.-P. (2022). *Graph attention network for link prediction of gene regulations from single-cell RNA-sequencing data*. Bioinformatics, 38(19), 4522–4529.  
  https://doi.org/10.1093/bioinformatics/btac559

- Zhang, K., Wang, C., Sun, L., & Zheng, J. (2022). *Prediction of gene co-expression from chromatin contacts with graph attention network*. Bioinformatics, 38(19), 4457–4465.  
  https://doi.org/10.1093/bioinformatics/btac535

- Yu, W., Lin, Z., Lan, M., & Ou-Yang, L. (2025). *GCLink: A graph contrastive link prediction framework for gene regulatory network inference*. Bioinformatics, 41(3).  
  https://doi.org/10.1093/bioinformatics/btaf074

- Narganes-Carlon, D., Myatt, A., Mudaliar, M., & Crowther, D. J. (2024). *GATher: Graph Attention Based Predictions of Gene-Disease Links*. arXiv:2409.16327  
  https://doi.org/10.48550/arXiv.2409.16327

---

## Biological Graph Learning & Generative Models

- Yu, T., Ekbote, C., Morozov, N., et al. (2025). *Tissue Reassembly with Generative AI*. Bioinformatics.  
  https://doi.org/10.1101/2025.02.13.638045

- Gjoni, K., Gunsalus, L. M., Kuang, S., et al. (2025). *Comparing chromatin contact maps at scale: Methods and insights*. Nature Methods, 22(4), 824–833.  
  https://doi.org/10.1038/s41592-025-02630-5

---

## Graph-Based Biological Applications

- Aamer, N., Asim, M. N., Vollmer, S., & Dengel, A. (n.d.). *An Explainable Knowledge Graph-Driven Approach to Decipher the Link Between Brain Disorders and the Gut Microbiome*.

- Dip, S. A., & Zhang, L. (n.d.). *Predicting Unseen Gene Perturbation Response Using Graph Neural Networks with Biological Priors*.

- Yuan, X. (n.d.). *Graph neural networks for spatial gene expression analysis of the developing human heart*.

---

## Tools for Cellular Analysis

- Stirling, D. R., Swain-Bowden, M. J., Lucas, A. M., et al. (2021). *CellProfiler 4: Improvements in speed, utility and usability*. BMC Bioinformatics, 22(1), 433.  
  https://doi.org/10.1186/s12859-021-04344-9

---

---

# 🌐 External References

- Spatial Transcriptomics Overview  
  https://www.nature.com/articles/s41592-021-01109-1

- Graph Neural Networks Survey  
  https://arxiv.org/abs/1812.08434

- Morphometrics in Biology  
  https://doi.org/10.1038/nmeth.2930

- Xenium Platform (10x Genomics)  
  https://www.10xgenomics.com/products/xenium-in-situ

---

---
# 📬 Contact

---

---
# 🤝 Acknowledgements

- Spatial transcriptomics datasets (10x Genomics Xenium 🧬)
- Costa Lab
- HPC infrastructure support
- Project supervisors and collaborators

### LUNA compatibility patch

After cloning or reinitializing `external/LUNA`, apply the local compatibility patch:

```bash
git -C external/LUNA apply --check ../../luna-compat-fixes.patch
git -C external/LUNA apply ../../luna-compat-fixes.patch
````

The patch removes the Scanpy dependency from the AnnData helper and handles
zero-range coordinates in LUNA inference/test datasets.
