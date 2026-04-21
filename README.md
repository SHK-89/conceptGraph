# Linking Scene Graph Generation Models with Human Visual Attention Graphs

This repository contains the implementation for my lab rotation project on linking open-vocabulary scene graph representations with human visual attention in dynamic scenes.

## Project Overview

The goal of this project is to investigate which semantic relation types in dynamic scenes are preferentially selected by human gaze.

The full pipeline combines:
1. **Scene graph construction** from dynamic videos using object annotations and a vision-language model
2. **Temporal relation extraction** from human gaze scanpaths
3. **Predicate embedding and clustering** into semantic relation families
4. **Cluster-level comparison** between scene availability and gaze-based selection

The central research question is:

> Which semantic relation clusters in an open-vocabulary scene graph are preferentially selected by human gaze in dynamic environments?

---

## Repository Structure

The `master` branch currently includes the following main components: `scene_graph`, `embedding`, `clustering`, and `cluster_analyzer`. :contentReference[oaicite:1]{index=1}

A brief overview of the folders:

- `scene_graph/`  
  Scene graph generation pipeline and graph construction code

- `embedding/`  
  Code for embedding unique predicates into 300-dimenstion vector space using GloVe.
  
- `clustering/`  
  Code for clustering using K-medoids.

- `cluster_analyzer/`  
  Code for the last step of the pipeline to compare the whole predicate cluster with the cluster preferred by gaze.

---

## Methodological Pipeline

### 1. Scene Graph Construction
For each frame of a video:
- objects are obtained from the UVO-based annotations
- object crops, bounding boxes, and centers are extracted
- pairwise object relations are inferred using a vision-language model
- frame-wise scene graphs are stored in structured JSON format

### 2. Temporal Relation Extraction
From the eye-tracking data:
- only foveation events (`FOV`) are used
- attended objects are identified for each foveation
- scanpaths are constructed across consecutive foveations
- relations between consecutively attended objects are extracted from the scene graph

### 3. Predicate Cleaning, Embedding, and Clustering
- lexical variants of predicates are cleaned and normalized
- unique predicates are embedded in a semantic vector space
- cosine distance is used for predicate similarity
- K-medoids clustering groups predicates into interpretable semantic families

### 4. Cluster-Level Attention Analysis
Two distributions are compared:
- all scene-graph relations
- gaze-selected temporal relations

This makes it possible to distinguish between:
- relations that are merely frequent in the environment
- relations that are preferentially selected by human attention

---

## Python Environment

The pipeline was implemented primarily in **Python 3.13**.

Because of dependency compatibility for the clustering stage, the **clustering analysis was run separately in Python 3.11**.

---

## Reproducibility

This repository contains the code used for:
- scene graph generation
- temporal relation extraction
- predicate embedding
- clustering
- cluster-level attention analysis
- visualization of results

---

## Status

This repository is part of my lab rotation project and is under active development.

---

## Author

**Shokoofehsadat Hosseini**

Lab rotation project on scene graph generation and human visual attention at SCIoI project 57.
https://www.scienceofintelligence.de/research-projects/project_57/
