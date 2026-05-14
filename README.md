# Enhancing Spatial Transcriptomics with Morphometrics for Cellular Neighbor Prediction

## Project Overview

This project investigates whether cellular neighborhood relationships can be predicted using:

- gene expression,
- morphometric features,
- or a combination of both.

The problem is framed as a graph-based link prediction task where:
- cells are nodes,
- neighboring relationships are edges,
- and biological features are used for prediction.

---

# Research Question

## Main Question

Can cellular neighborhood relationships be predicted from gene expression and/or morphometric features?

---

## Subquestions

- Does morphology improve prediction accuracy?
- Which modality contributes more strongly?
- Are transcriptomic and morphometric features complementary?
- Can graph-based methods outperform classical nearest-neighbor methods?
- Can graph attention reveal biologically meaningful interactions?

---

# Repository Structure

```text
project-root/
├── docs/
├── data/
├── notebooks/
├── scripts/
├── results/
├── models/
└── references/
```

---

# Literature Review

## Spatial Transcriptomics

### Summary
Spatial transcriptomics preserves spatial tissue organization while measuring gene expression.

### Important Concepts
- tissue architecture
- cellular neighborhoods
- spatial embeddings
- spatial graphs

### Relevant Technologies
- Visium
- MERFISH
- Slide-seq
- seqFISH

### Notes
- Add findings from reviewed papers here.

---

## Morphometrics

### Summary
Morphometrics studies quantitative cellular shape and structural properties.

### Candidate Features
- area
- perimeter
- eccentricity
- compactness
- texture
- density

### Notes
- Add observations and useful methods here.

---

## Graph Learning

### Summary
Graph learning models biological systems as relational structures.

### Relevant Methods
- kNN
- weighted kNN
- Graph Convolutional Networks (GCN)
- Graph Attention Networks (GAT)
- link prediction

### Notes
- Add graph-learning related literature here.

---

# Paper Review Template

## Paper Title

### Citation
Authors, year, journal/conference.

### Main Idea
Short summary of the paper.

### Methodology
Describe the computational approach used.

### Dataset
What biological dataset was used?

### Strengths
- 
- 
- 

### Weaknesses
- 
- 
- 

### Relevance to Project
Why is this paper useful for our work?

### Ideas to Reuse
- 
- 
- 

---

# Proposed Methodology

## Data Collection

### Datasets
| Dataset | Status | Notes |
|---|---|---|
|  |  |  |

---

## Preprocessing

- [ ] Normalize gene expression
- [ ] Clean metadata
- [ ] Construct spatial graphs
- [ ] Extract morphology features
- [ ] Generate train/test splits

---

## Feature Engineering

### Transcriptomic Features
- PCA embeddings
- marker genes
- normalized counts

### Morphometric Features
- shape descriptors
- density features
- texture metrics
- graph topology

---

# Models

## Baseline Models

- [ ] vanilla kNN
- [ ] weighted kNN
- [ ] morphology-only baseline
- [ ] transcriptomics-only baseline

---

## Advanced Models

- [ ] GCN
- [ ] GAT
- [ ] multimodal graph learning

---

# Experimental Design

## Benchmark Table

| Model | Features | Accuracy | F1 | Notes |
|---|---|---|---|---|
| Baseline kNN | Expression |  |  |  |

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Runtime
- Memory usage

---

# Results

## Current Findings

### Observations
- 

### Failed Approaches
- 

### Successful Approaches
- 

---

# Biological Interpretation

## Key Questions

- Which cells cluster together?
- Does morphology correlate with spatial proximity?
- Are graph attention weights biologically meaningful?

---

# Current Tasks

## High Priority

- [ ] Download datasets
- [ ] Run dump-scripts
- [ ] Reproduce baseline
- [ ] Implement weighted kNN

---

## Medium Priority

- [ ] Add morphology features
- [ ] Build graph pipelines
- [ ] Benchmark models

---

## Long-Term Goals

- [ ] GAT implementation
- [ ] Ablation studies
- [ ] Explainability analysis
- [ ] Final presentation

---

# Timeline

## 4.5 – 29.6
Project development

### Goals
- dataset setup
- baseline implementation
- morphology integration
- benchmarking
- graph learning experiments

---

## 6.7
Project Presentation

### Deliverables
- presentation slides
- GitHub/GitLab repository
- experimental results
- benchmarking analysis

---

# Discussion and Future Work

## Potential Future Directions

- multimodal transformers
- contrastive learning
- explainable graph attention
- tissue reconstruction
- scalable graph learning

---

# References

## Papers

- Add references here

---

# Notes

## Meeting Notes

### Date

### Discussion

### Action Items
