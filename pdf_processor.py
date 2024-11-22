import streamlit as st
import os
from typing import List, Dict
from pathlib import Path
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import tempfile

# Cấu hình Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

class GoogleDriveHelper:
    def __init__(self):
        self.creds = None
        
    def authenticate(self):
        """Xác thực với Google Drive"""
        # Sử dụng st.secrets để lấy credentials từ Streamlit
        credentials_dict = {
            "installed": {
                "client_id": st.secrets["client_id"],
                "client_secret": st.secrets["client_secret"],
                "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"]
            }
        }
        
        flow = InstalledAppFlow.from_client_config(credentials_dict, SCOPES)
        self.creds = flow.run_local_server(port=0)
        
    def get_pdf_files(self, folder_id):
        """Lấy danh sách file PDF từ thư mục Google Drive"""
        service = build('drive', 'v3', credentials=self.creds)
        
        query = f"'{folder_id}' in parents and mimeType='application/pdf'"
        results = service.files().list(
            q=query,
            fields="files(id, name)"
        ).execute()
        
        return results.get('files', [])
        
    def download_file(self, file_id):
        """Tải file từ Google Drive"""
        service = build('drive', 'v3', credentials=self.creds)
        request = service.files().get_media(fileId=file_id)
        
        file = io.BytesIO()
        downloader = MediaIoBaseDownload(file, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        
        return file

class E5MultilingualEmbeddings:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
    def average_pool(self, last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _get_embedding(self, text: str) -> List[float]:
        text = f"passage: {text}"
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=512,
                              return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings[0].cpu().numpy().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        text = f"query: {text}"
        return self._get_embedding(text)

class PDFProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        self.embeddings = E5MultilingualEmbeddings()
        
        # Khởi tạo Pinecone với cú pháp mới
        pc = Pinecone(api_key=pinecone_api_key)
        
        # Kiểm tra xem index đã tồn tại chưa
        try:
            # Thử lấy index hiện có
            self.index = pc.Index(index_name)
        except:
            # Nếu index chưa tồn tại, tạo mới
            pc.create_index(
                name=index_name,
                dimension=1024,  # dimension cho E5-large
                metric='cosine',
                spec=ServerlessSpec(
                    cloud=pinecone_environment.split('-')[2],  # 'aws' hoặc 'gcp'
                    region=pinecone_environment.split('-')[0]  # region
                )
            )
            self.index = pc.Index(index_name)

    def extract_text_from_pdf(self, pdf_content: bytes) -> List[Dict]:
        doc = fitz.open(stream=pdf_content, filetype="pdf")
        chunks = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            headers = [block for block in page.get_text("dict")["blocks"] 
                      if block.get("lines") and len(block["lines"][0]["spans"]) > 0 
                      and block["lines"][0]["spans"][0]["size"] > 12]
            
            chunks.append({
                "text": text,
                "metadata": {
                    "page": page_num + 1,
                    "headers": [h["lines"][0]["spans"][0]["text"] for h in headers]
                }
            })
        
        return chunks

    def create_semantic_chunks(self, text_chunks: List[Dict]) -> List[Dict]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        semantic_chunks = []
        for chunk in text_chunks:
            splits = text_splitter.create_documents(
                texts=[chunk["text"]], 
                metadatas=[chunk["metadata"]]
            )
            semantic_chunks.extend(splits)
        
        return semantic_chunks

    def upload_to_pinecone(self, chunks: List[Dict], file_name: str, batch_size: int = 100):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]
            
            embeddings = self.embeddings.embed_documents(texts)
            
            vectors = []
            for j, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                metadata['source'] = file_name
                vectors.append({
                    "id": f"{file_name}_chunk_{i+j}",
                    "values": embedding,
                    "metadata": {
                        "text": text,
                        **metadata
                    }
                })
            
            self.index.upsert(vectors=vectors)

    def process_file(self, file_content: bytes, file_name: str):
        text_chunks = self.extract_text_from_pdf(file_content)
        semantic_chunks = self.create_semantic_chunks(text_chunks)
        self.upload_to_pinecone(semantic_chunks, file_name)

    def query_documents(self, query: str, top_k: int = 5):
        query_embedding = self.embeddings.embed_query(query)
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results

def main():
    st.title("PDF Document Processor with Google Drive")
    
    # Khởi tạo session state
    if 'drive_helper' not in st.session_state:
        st.session_state.drive_helper = GoogleDriveHelper()
        
    if 'processor' not in st.session_state:
        try:
            st.session_state.processor = PDFProcessor(
                pinecone_api_key=st.secrets["pinecone_api_key"],
                pinecone_environment=st.secrets["pinecone_environment"],
                index_name=st.secrets["pinecone_index_name"]
            )
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return
    
    # Thêm Google Drive folder ID
    folder_id = st.text_input("Enter Google Drive Folder ID")
    
    if st.button("Authenticate & Process"):
        try:
            with st.spinner("Authenticating with Google Drive..."):
                st.session_state.drive_helper.authenticate()
            
            with st.spinner("Getting PDF files..."):
                pdf_files = st.session_state.drive_helper.get_pdf_files(folder_id)
            
            if not pdf_files:
                st.warning("No PDF files found in the specified folder")
                return
                
            progress_bar = st.progress(0)
            for i, file in enumerate(pdf_files):
                with st.spinner(f"Processing {file['name']}..."):
                    file_content = st.session_state.drive_helper.download_file(file['id'])
                    st.session_state.processor.process_file(
                        file_content.getvalue(),
                        file['name']
                    )
                progress_bar.progress((i + 1) / len(pdf_files))
            
            st.success("All files processed successfully!")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    # Phần tìm kiếm
    st.header("Search Documents")
    query = st.text_input("Enter your search query")
    
    if query:
        with st.spinner("Searching..."):
            results = st.session_state.processor.query_documents(query)
            
            for match in results.matches:
                st.markdown("---")
                st.markdown(f"**Score:** {match.score:.2f}")
                st.markdown(f"**Source:** {match.metadata['source']}")
                st.markdown(f"**Page:** {match.metadata['page']}")
                st.markdown(f"**Text:**\n{match.metadata['text']}")

if __name__ == "__main__":
    main()
