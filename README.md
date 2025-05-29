# 📰 Fake News Image-Based Detection

A deep learning-powered application that detects fake news based on uploaded images by verifying their authenticity, extracting relevant textual content, and cross-checking the information with real-time search engine results.

## 🚀 Features

- 🖼️ Upload an image to verify if it has been manipulated or generated.
- 🧠 Detect fake or altered visuals using **Generative Adversarial Networks (GANs)**.
- 🔍 Extract or generate text from images using **OCR** and analyze the content using **NLP**.
- 🌐 Cross-verify extracted text against **Google's top 5 search results** to detect inconsistencies.
- 📊 Display results in a clear and interactive **Streamlit** interface.

## 🛠️ Tech Stack

- **Frontend/UI:** Streamlit
- **Languages:** Python
- **Deep Learning:** GANs, CNN
- **Text Extraction:** OCR (Tesseract)
- **Text Analysis:** NLP (NLTK / SpaCy)
- **Web Verification:** Google Search API / SerpAPI
- **Others:** Requests, BeautifulSoup (for scraping if applicable), PIL, OpenCV

## 📸 Screenshots

![image](https://github.com/user-attachments/assets/7091d8c2-aa1c-4a4d-a6bc-6e75a47f3fae)


## 🧪 How It Works

1. **Image Upload:** User uploads an image suspected of being fake.
2. **Authenticity Check:** GAN model checks for signs of image manipulation.
3. **Text Extraction:** OCR extracts visible text from the image.
4. **NLP Analysis:** NLP techniques clean and process the extracted text.
5. **Fact-Checking:** Top 5 search results for the text are fetched and compared.
6. **Result:** Displays whether the news/image is likely real or fake based on all checks.

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/fake-news-image-detector.git
   cd fake-news-image-detector
   ```

2. **Create a virtual environment (optional):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    Run the Streamlit app:
    ```
4. **Run the streamlit:**
    ```bash
    streamlit run app.py
    ```
