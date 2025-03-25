# Data Scientist’s Learning Assistant
---
The Data Scientist’s Learning Assistant is an innovative multimodal chatbot designed to enhance the educational experience in Data Science, Artificial Intelligence, and Machine Learning. By leveraging cutting-edge Generative AI and Natural Language Processing (NLP) techniques, the assistant delivers personalized and context-aware responses to user queries. It integrates diverse educational resources—including textbooks, research papers, and educational videos—to create an interactive learning environment that bridges theoretical knowledge and practical application.

---

## Objectives
- **Enhance Learning:** Provide accurate, context-aware answers to help students, early career professionals, and researchers master complex data science topics.
- **Integrate Multimodal Data:** Combine text and video data to offer a comprehensive knowledge base.
- **Develop Advanced Models:** Explore and compare multiple model architectures to improve question answering (QA) capabilities.
- **Interactive Interface:** Deliver real-time, user-friendly interactions via a web interface built with Flask and Gradio.

---

## Methodology

### 1. Data Collection
- **Textual Data:**  
  - Collected 55 textbooks and relevant research papers from Google and various online libraries.
- **Video Data:**  
  - Downloaded 44 educational videos from the Codebasics YouTube channel.
  
These diverse sources ensure a broad and rich knowledge base for the assistant.

### 2. Data Transformation & Preprocessing
- **Text Data Processing:**
  - **Extraction:** Utilized tools such as `pdfplumber` and `pytesseract` to extract text from PDFs.
  - **Cleaning:** Removed unwanted characters, special symbols, and metadata.
  - **Tokenization & Stopword Removal:** Employed NLTK for breaking text into tokens and filtering out common stopwords.
  - **Saving:** Stored cleaned text as JSON for downstream processing.
  
- **Video Data Processing:**
  - **Audio Extraction:** Used `moviepy` and `ffmpeg` to extract audio from videos.
  - **Transcription:** Leveraged OpenAI’s Whisper to transcribe audio into text.
  - **Saving:** Saved transcripts in JSON format for later use.

### 3. Modeling
The project explores three sequential models with increasing complexity:

#### Model 1: BERT-based Question Answering System
- **Architecture:**  
  Built using PyTorch and Hugging Face’s Transformers (BertTokenizer and BertForQuestionAnswering).
- **Training:**  
  Data was processed in small batches (batch size = 2) to manage limited hardware (MacBook Air M3). Training spanned several epochs:
  - **Epoch 1:** Initial rapid learning with loss dropping from 6.44 to near 0.005.
  - **Epoch 2:** Fine-tuning with losses stabilizing around 0.002–0.003.
  - **Epoch 3:** Refinement achieving an average loss as low as 0.0017.
- **Evaluation Metrics:**  
  - **Exact Match (EM):** 100%
  - **F1 Score:** 1.00  
  (Indicating near-perfect performance on controlled test sets.)

#### Model 2: Advanced Transformer-based System
- **Improvements:**  
  Focused on dynamic adjustments using AutoTokenizer and AutoModel to handle larger and more complex datasets.
- **Challenges:**  
  Despite sophisticated embedding and similarity computations, Model 2 struggled—resulting in 0% EM and F1 scores.

#### Model 3: Optimized QA System with Interactive Interface
- **Enhancements:**  
  Combined advanced NLP libraries (Transformers, Sentence-Transformers) with an interactive front-end built with Flask and Gradio.
- **Strengths & Limitations:**  
  - **Fast Context Retrieval:** Retrieval speeds as low as 0.08 seconds.
  - **Accuracy Issues:** Still faces challenges in precisely matching ground truth responses.
  
### 4. Front-End Interface
- **Implementation:**  
  A Flask-based web application coupled with Gradio provides a user-friendly interface. Users can submit questions through an HTML form, which are then processed by the QA system to retrieve relevant context and generate responses in real time.

---

## Model Evaluation & Results

- **Model 1 (BERT-based QA):**  
  Outstanding performance in controlled settings with perfect EM and F1 scores. However, potential overfitting and hardware limitations were noted.
  
- **Model 2 (Advanced Transformer):**  
  Showed limitations in embedding quality and retrieval mechanisms, resulting in 0% accuracy on key metrics.
  
- **Model 3 (Optimized QA with Interactive Interface):**  
  Demonstrates excellent processing speeds and interactive capabilities but requires further refinement for accurate answer generation.

**Key Takeaways:**
- **Data Quality & Preprocessing:** Rigorous preprocessing of text and video data is essential for model performance.
- **Model Progression:** While Model 1 excels in accuracy under controlled conditions, Models 2 and 3 reveal challenges in scaling and real-world performance.
- **Interactive Systems:** Integrating a responsive front-end (Flask and Gradio) is vital for practical application but must be coupled with robust backend accuracy.

---

## Deliverables

- **Hardware:**  
  MacBook Air M3, 8 GB RAM

- **Software & Libraries:**  
  - Anaconda Navigator, Jupyter Notebook
  - PyPDF2, pdfminer.six, nltk, moviepy, ffmpeg-python, OpenAI Whisper
  - PyTorch, Transformers, Sentence-Transformers, Flask, Gradio
  - Plus additional libraries (pandas, numpy, scikit-learn, etc.)

- **Dataset:**  
  Curated and preprocessed collections from textbooks, PDFs, and educational videos, stored in JSON format.

- **Code Files:**
  - `Text Data Preprocessing.ipynb`
  - `Video Data Preprocessing.ipynb`
  - Other model training and interface notebooks

 ## How to Run
1. **Clone the Repository:**  
   Download or clone the repository from GitHub: [Data-Scientist-s-Learning-Assistant](https://github.com/SomitaChaudhari/Data-Scientist-s-Learning-Assistant)
2. **Set Up Environment:**  
   Use Anaconda or a virtual environment and install the required libraries (see Deliverables section).
3. **Run Notebooks:**  
   Execute the Jupyter Notebooks (e.g., `Text Data Preprocessing.ipynb`, `Video Data Preprocessing.ipynb`) sequentially to preprocess data and train models.
4. **Start the Interface:**  
   Run the Flask/Gradio application to interact with the QA system in real time.
5. **Explore & Evaluate:**  
   Review logs, evaluation metrics, and interactive outputs to understand model performance and system behavior.
