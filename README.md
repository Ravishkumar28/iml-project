# Visual Question Answering (VQA) using Cross-Attention and Transformers

## 🧠 Problem Statement
This project addresses **Visual Question Answering (VQA)** — a task where a model takes an image and a natural language question as input, and outputs a **yes/no** answer based on spatial and logical relationships in the image.

Example questions:
- "Is there a red shape above a circle?"
- "Is a red shape blue?"

---

## 📊 Model Overview

The model is a multi-component deep learning architecture trained end-to-end to classify image-question pairs.

### 🔧 Components
1. **ResNet Feature Extractor**
   - Pretrained `ResNet50` backbone extracts image features.
   - Outputs a fixed-size representation via linear projection.

2. **BERT Encoder**
   - Encodes natural language questions using `BERT-base-uncased`.
   - Projects to match image feature dimensions.

3. **Cross-Attention Module**
   - Fuses image and question features using cross-attention.
   - The question serves as a query to attend to image features.

4. **Attention Mechanism**
   - Highlights relevant image parts via attention weights.

5. **Classifier**
   - A feedforward neural network predicts a binary (yes/no) answer.

---

## 🧮 Mathematical Formulation

- Image feature: `v = Wv * ResNet(x) + bv`
- Question feature: `q = Wq * BERT(q) + bq`
- Cross-Attention: `a = softmax(qkᵀ / √d), c = av`
- Final classification: `y = Wc * a + bc`
- Loss function: Cross-entropy loss over predictions and ground-truth labels

---

## 📦 Code Components

### 📁 Main Classes
- `ResnetFeatureExtractor`: Extracts image features.
- `CrossAttention`: Implements dot-product attention between image and text.
- `VQAModel`: Combines all modules into a single forward pipeline.
- `ShapeGenerator`: Creates synthetic images, questions, and labels with optional noise.
- `ShapesDataset`: Loads synthetic data with augmentations and splits.

### 🛠 Key Functions
- `preprocess_image()`: Image resizing, normalization, tensor conversion.
- `preprocess_question()`: BERT tokenizer-based padding and truncation.
- `train_one_epoch()`: One training epoch with backprop and metric tracking.
- `validate()`: Evaluates model performance on validation data.

---

## 📈 Results

### Training Observations
- Achieved **78.5% accuracy** over 6 epochs initially.
- Observed **overfitting** due to small test dataset (200 images, 65 noisy).
- Training dataset: 1000 images, 317 with noise.

### Noise Experiments
- Maximum accuracy:
  - **TV noise (25%)**: 85%
  - **Shape noise (1%)**: 89%
  - **Color noise (1%)**: 90%
- Combined setting: TV=25%, Shape=1%, Color=1%
  - **Test Accuracy:** 96.5%
  - **Train Accuracy:** 99.5%

---

## 🔭 Future Scope
- Improve generalization via regularization and better preprocessing.
- Expand dataset with complex scenes and multi-object relationships.
- Build an interactive chatbot-style app to input images and tune noise levels dynamically.

---

## 👤 Contributor
- **Ravish Kumar** (B23ME1053)

---


