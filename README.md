# Enhancing Spatial Transcriptomics with Morphometrics for Cellular Neighbor Prediction

---

# 1. Introduction

## Problem Statement

Can we predict cellular neighborhood relationships using:
- gene expression,
- morphometric features,
- or both combined?

---

## Motivation

Why is this biologically/computationally important?

- 
- 
- 

---

## Objectives

- [ ] Predict cell neighbors
- [ ] Compare transcriptomics vs morphology
- [ ] Benchmark classical and graph methods
- [ ] Analyze biological interpretability

---

# 2. Background

## Spatial Transcriptomics

### Key Concepts
- 
- 
- 

### Challenges
- 
- 
- 

---

## Morphometrics

### Key Concepts
- 
- 
- 

### Candidate Features
- 
- 
- 

---

## Graph Learning

### Key Concepts
- 
- 
- 

### Relevant Methods
- kNN
- weighted kNN
- GCN
- GAT
- link prediction

---

# 3. Literature Review

## Spatial Transcriptomics Papers

### Paper

#### Citation

#### Main Contribution

#### Relevance

#### Notes

---

## Morphometrics Papers

### Paper

#### Citation

#### Main Contribution

#### Relevance

#### Notes

---

## Graph Learning Papers

### Paper

#### Citation

#### Main Contribution

#### Relevance

#### Notes

---

# 4. Dataset and Data Processing

## Datasets

| Dataset | Description | Status | Notes |
|---|---|---|---|
|  |  |  |  |

---

## Preprocessing Pipeline

- [ ] Download datasets
- [ ] Clean metadata
- [ ] Normalize expression
- [ ] Extract morphology
- [ ] Build spatial graphs
- [ ] Generate train/test splits

---

## Spatial Graph Construction

### Graph Definition
- radius graph
- k-nearest spatial graph
- Delaunay triangulation

### Notes
- 
- 
- 

---

# 5. Feature Engineering

## Transcriptomic Features

### Methods
- PCA
- highly variable genes
- embeddings

### Notes
- 
- 
- 

---

## Morphometric Features

### Features
- area
- perimeter
- eccentricity
- texture
- density

### Notes
- 
- 
- 

---

## Combined Features

### Fusion Strategy
- concatenation
- weighted fusion
- learned embeddings

### Notes
- 
- 
- 

---

# 6. Models

## Baseline Models

### Vanilla kNN

#### Description

#### Results

---

### Weighted kNN

#### Weighting Methods
- 1/d
- 1/d²
- Gaussian weighting

#### Results

---

## Graph Models

### GCN

#### Architecture

#### Results

---

### GAT

#### Architecture

#### Attention Interpretation

#### Results

---

# 7. Link Prediction

## Problem Formulation

### Nodes
cells

### Edges
neighbor relationships

### Prediction Task
predict whether two cells should share an edge

---

## Negative Sampling

### Strategy
- 
- 
- 

---

# 8. Experimental Design

## Benchmarking

| Model | Features | Accuracy | F1 | ROC-AUC | Notes |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

---

## Ablation Studies

- [ ] transcriptomics only
- [ ] morphology only
- [ ] combined features
- [ ] without attention
- [ ] different graph construction methods

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Runtime

---

# 9. Biological Interpretation

## Questions

- Which cells are easiest to predict?
- Does morphology improve neighborhood prediction?
- Are attention weights biologically meaningful?
- Which features contribute most strongly?

---

## Visualization Ideas

- spatial graphs
- predicted neighbors
- attention maps
- embedding visualizations

---

# 10. Current Results

## Working Approaches

- 
- 
- 

---

## Failed Approaches

- 
- 
- 

---

## Observations

- 
- 
- 

---

# 11. Discussion

## Challenges

- noisy data
- sparse expression
- graph sensitivity
- feature imbalance

---

## Limitations

- 
- 
- 

---

# 12. Future Work

- multimodal transformers
- explainable graph learning
- contrastive learning
- larger spatial datasets

---

# 13. Task Tracking

## Current Tasks

- [ ] Set up repository
- [ ] Download datasets
- [ ] Run dump-scripts
- [ ] Reproduce baseline
- [ ] Implement weighted kNN
- [ ] Add morphology features
- [ ] Benchmark models
- [ ] Prepare presentation

---

# 14. Timeline

| Date | Milestone | Status |
|---|---|---|
| 4.5 | Project Start | Complete |
| 29.6 | Development Complete | Pending |
| 6.7 | Final Presentation | Pending |

---

# 15. Presentation Structure

## Sections

- Introduction
- Biological Motivation
- Methodology
- Model Design
- Benchmarking
- Results
- Biological Interpretation
- Discussion
- Future Work

---

# 16. References

## Papers

- 

---

# 17. Meeting Notes

## Date

### Discussion

### Decisions

### Action Items

- 
- 
- 
