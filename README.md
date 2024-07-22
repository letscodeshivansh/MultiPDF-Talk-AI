# [MultiPDF-Talk ](https://multipdf-talk.streamlit.app/)

The MultiPDF Chat Application is a web-based tool that allows users to interact with the content of multiple PDF files through a conversational interface. Users can upload PDFs, ask questions about the content, and receive detailed answers. This application leverages the power of Google Generative AI for natural language processing and FAISS for efficient vector search.

## Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Libraries**:
  - Streamlit: For building the web interface
  - PyPDF2: For extracting text from PDFs
  - Langchain: For text processing and question-answering chains
  - Google Generative AI: For natural language understanding and response generation
  - FAISS: For fast and efficient similarity search
  - dotenv: For environment variable management
  - logging: For error and process logging

## Installation 💻

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/letscodeshivansh/MultiPDF-Talk-AI.git

2. Create a Virtual Env (recommended):

    ```bash
    conda create -p venv python==3.12
    conda activate venv/
    ```

3. Install python libraries:

    ```bash
    pip install -r requirements.txt
    ```

4. Set Up Environment Variables:
   
    ```bash
    GOOGLE_API_KEY=your_google_api_key_here
    ```
    
5. Start the project:

    ```bash
    streamlit run app.py 
    ```

6. Now, upload your PDFs and start interacting- 

## Acknowledgements:

- [Streamlit](https://streamlit.io/) for providing a powerful and easy-to-use framework for building web applications.
- [Google Generative AI](https://ai.google/) for the natural language processing capabilities.
- [FAISS](https://github.com/facebookresearch/faiss) for the efficient similarity search.
