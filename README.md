---
title: Comparative Analysis of Text Topic Classification Approaches
---

# Comparative Analysis of Text Topic Classification Approaches

A comparative analysis of four different approaches to text topic classification, aiming to classify a text into one of the following categories: **World**, **Sports**, **Business**, or **Sci/Tech**.

## Approaches

The following approaches were implemented and compared:

1. **SpaCy Zero-Shot Classification**
2. **Together AI with Chain of Thought Prompting**
3. **Together AI with Self-Ask Prompting**
4. **Together AI with Reversed Prompting** (a novel technique)

## Approach Descriptions

### 1. SpaCy Zero-Shot Classification
SpaCy’s zero-shot classification leverages pre-trained language models to classify texts without the need for additional training.

### 2. Together AI with Chain of Thought Prompting
This approach guides the model step-by-step through its reasoning process, using Together AI, which feeds from different large language models (LLMs) and returns the best response.

### 3. Together AI with Self-Ask Prompting
The model generates and answers its own questions to reach a conclusion, leveraging Together AI.

### 4. Together AI with Reversed Prompting
This novel technique starts by thinking about the categories and then applies the best fitting category, using Together AI.

### Accuracy scores
- **SpaCy Zero-Shot Classification:** 0.67
- **Chain of Thought Prompting:** 0.87
- **Self-Ask Prompting:** 0.74
- **Reversed Prompting:** 0.80

