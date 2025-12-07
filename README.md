
# 🧠 Multi-Head Attention Visualizer

Interactive Tool to Explore Attention Mechanisms in Transformer Models

---

## 🚀 Overview

This project visualizes **multi-head self-attention** in Transformer-based deep learning models (BERT).
By selecting a **token, layer, and attention head**, the user can observe how the model distributes attention across the input sentence.

🔍 Useful for:

* Deep Learning & NLP learning
* Transformer interpretability
* Research & education
* Viva / academic projects

---

## 🌐 Live Demo

🔗 **App Link:** [https://multi-head-attention-output-viewer.streamlit.app/](https://multi-head-attention-output-viewer.streamlit.app/)
🔗 **GitHub Repository:** [https://github.com/thanushshetty9353/DeepLearningAssignement](https://github.com/thanushshetty9353/DeepLearningAssignement)

---

## 📸 Screenshots

### 🏠 Home View

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/8bac0a3f-45f5-403a-bc7b-eebe2b872bd9" />

### 🔥 Attention Heatmap

<img width="100%" alt="image" src="https://github.com/user-attachments/assets/be5e1159-6611-4a52-9f49-a4cb1974a536" />

### 🔍 All Heads Comparison

<img width="1919" height="975" alt="image" src="https://github.com/user-attachments/assets/dc2d131b-9103-46fa-b070-c4803989dfdc" />

---

## ✨ Features

| Feature                             | Status |
| ----------------------------------- | ------ |
| Token-level attention visualization | ✔      |
| Multi-head comparison               | ✔      |
| Heatmap attention matrix            | ✔      |
| Sentence-level visual highlighting  | ✔      |
| Layer + Head selection              | ✔      |
| Works for any sentence              | ✔      |

---

## 🛠️ Tech Stack

| Component     | Technology                       |
| ------------- | -------------------------------- |
| Language      | Python                           |
| Framework     | Streamlit                        |
| NLP Model     | BERT (Hugging Face Transformers) |
| Visualization | Plotly                           |
| Deployment    | Streamlit Cloud                  |

---

## 📂 Project Structure

```
📦 Project
┣ 📜 app.py
┣ 📜 attention_utils.py
┣ 📜 requirements.txt
┣ 📂 .streamlit
┃ ┗ 📜 config.toml
┗ 📜 README.md
```

---

## ⚙️ Installation & Running Locally

### 🔹 Clone the repository

```bash
git clone https://github.com/thanushshetty9353/DeepLearningAssignement.git
cd DeepLearningAssignement
```

### 🔹 Install dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run the application

```bash
streamlit run app.py
```

---

## 🧠 How it Works

1. Enter a sentence
2. Select **Layer**, **Head**, and **Token**
3. Visual output includes:

   * Sentence attention highlights
   * Full head heatmap
   * All-head comparison inside the layer
4. Understand how each attention head behaves differently

---

## 📚 Concepts Covered

* Transformers
* Self-Attention
* Multi-Head Attention
* NLP Model Interpretability
* Deep Learning Visualization

---

## 👨‍💻 Author

**💛 Thanush Shetty**

📍 India

🔗 GitHub: [https://github.com/thanushshetty9353](https://github.com/thanushshetty9353)

🔗 LinkedIn: [https://www.linkedin.com/in/thanush-shetty-a49801298/](https://www.linkedin.com/in/thanush-shetty-a49801298/)

---

## ⭐ Contributions

Contributions, issues, and feature requests are welcome!
If you like this project, don’t forget to **star ⭐ the repository**.

---

## 📄 License

This project is open-sourced under the **MIT License**.

---
