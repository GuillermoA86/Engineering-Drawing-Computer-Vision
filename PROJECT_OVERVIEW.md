# Project Overview — Engineering Drawing Computer Vision

## Objective

Build an end-to-end computer-vision system that recognizes engineering symbols from Piping and Instrumentation Diagram (P&ID) imagery.

## Real-world scenario

A process-engineering company maintains thousands of drawings. Engineers need to locate and inventory components such as valves and instruments. Manual review is slow and difficult to standardize.

The first stage of an automated workflow is symbol recognition.

## ML problem

Input:
- 100×100 engineering-symbol image

Output:
- symbol class
- probability/confidence
- top-k alternatives

## Dataset

SiED — Symbols in Engineering Drawings:
- 2,432 symbol instances
- extracted from P&IDs
- highly imbalanced
- public research dataset

## Model

ResNet-18 transfer learning.

## Evaluation

Primary:
- Macro F1
- Balanced Accuracy

Secondary:
- Accuracy
- Weighted F1
- Top-3 accuracy
- Confusion matrix

## Main risks

- class imbalance
- visually similar symbols
- small image resolution
- annotation ambiguity
- rotation and line-thickness variation
- train/test leakage through near-duplicate symbols

## Production extension

Full drawing → detector → symbol classifier → OCR → object/text association → structured P&ID graph → QA/revision analysis.
