import streamlit as st
from pypdf import PdfReader #convert whole pdf as reading state and make into one whole text 
from langchain_text_splitters import RecursiveCharacterTextSplitter   # covert whole text into chunks
from sentence_transformers import SentenceTransformer # CHUNKS to Numerical vectors ( embeddings )
import faiss  ## Building our FAISS Vector data base 
import numpy as np # convertring our embedding to an array 
from rank_bm25 import BM25Okapi # Add this BM25 keyword retrival and join with Faiss for best retrivels
from sentence_transformers import CrossEncoder # this cross encoder does reranking the exisiting RRF rannk
import os
from google import genai 
 

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedding_model, cross_encoder



st.set_page_config(
    page_title="AI PDF Research Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI PDF Research Assistant")

st.write("Ask questions about your PDF documents using RAG.")

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

 
if uploaded_file is not None:

    try:
        reader = PdfReader(uploaded_file)
        pdf_text = ""

        for page in reader.pages:
            page_text = page.extract_text() or ""
            pdf_text += page_text + "\n"

    except Exception:
        st.error("Unable to read this PDF. Please upload a valid, unprotected PDF.")
        st.stop()


    if not pdf_text.strip():
        st.warning("No readable text found in this PDF.")
        st.stop()

    st.write("Total Pages:", len(reader.pages))

    
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(pdf_text)
    
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    
    embedding_model , cross_encoder = load_models()
    
    
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        st.error("GEMINI_API_KEY is missing. Please configure your API key.")
        st.stop()

    gemini_client = genai.Client(api_key=api_key)

    
    
    chunk_embeddings = embedding_model.encode(chunks)
    
    
    chunk_embeddings = np.asarray(chunk_embeddings,dtype = "float32")
    faiss.normalize_L2(chunk_embeddings) 
    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(chunk_embeddings)
   
    
    
    st.success("PDF uploaded successfully")
    st.write("Filename:", uploaded_file.name)



question = st.text_input("Ask a question about your document")

if st.button("Get Answer"):
    
    if uploaded_file is None or not question.strip():
        st.warning("Please upload a PDF and enter a question.")
        st.stop()
           
    question_embedding = embedding_model.encode([question])
    question_embedding = np.asarray(question_embedding,dtype="float32")
       
    faiss.normalize_L2(question_embedding)
    top_k = min(10,len(chunks))
    scores,indices = faiss_index.search(question_embedding,top_k)
         
    query_tokens = question.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:10]
    
    faiss_ranks = indices[0].tolist()
    bm25_ranks = bm25_top_indices.tolist()
       
       
    def calculate_rrf(rank, k = 60):
        return 1/(k+rank)
       
    rrf_scores = {}
       
    for rank,chunk_id in enumerate (faiss_ranks,start=1):
        rrf_scores[chunk_id] = calculate_rrf(rank)
           
       
    for rank,chunk_id in enumerate (bm25_ranks,start =1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id,0) + calculate_rrf(rank)
           
           
    sorted_rrf = sorted(
        rrf_scores.items(),
        key = lambda item: item[1],
        reverse = True
    )
          
         
    #  Prepare question-chunk pairs from the top 10 RRF candidates
    
    cross_encoder_pairs = []
    
    for chunk_id , score in sorted_rrf[:10] :
        cross_encoder_pairs.append([question,chunks[chunk_id]])
        
    # Score each candidate based on its relevance to the user's question
        
    cross_encoder_scores = cross_encoder.predict(cross_encoder_pairs)
    
    
    # Sort candidate positions by Cross-Encoder score (highest first)
    
    reranked_indices = np.argsort(cross_encoder_scores)[::-1]
     
   
    
    # Build Gemini context using the top 3 Cross-Encoder ranked chunks
    # Include source numbers and original chunk IDs for citations
  
    
    context_parts = []
    
    for i , position in enumerate(reranked_indices[:3], start =1):
        chunk_id = sorted_rrf[int(position)][0]
        chunk_text = chunks[chunk_id]
        
        context_parts.append(
            f"[Source{i} | chunk ID :  {chunk_id}]\n {chunk_text}"
        )
        
    context = "\n\n".join(context_parts)
         
    
    prompt = f"""
    
    You aree an AI research assistant
    
    Answer the question using only the provided PDF context
    
    
    Instructions :
    
    1. Use only information available in the context
    2. If the answer is not available, respond :
        "I could not find the answer in the uploaded PDF."
    
    3. Cite the supporting source number , for example [Source 1].
    4. keep the answer clear and relevant to the question
    
    
    context :
    {context}
    
    Question :
    {question}
    
    Answer : 

    
    """
    
    try:
        with st.spinner("🤖 Gemini is generating your answer..."):
                response = gemini_client.models.generate_content(
                        model = "gemini-3-flash-preview",
                        contents = prompt
                        
                    )
            
        st.subheader("🤖 AI Research Assistant Answer")
        if response.text and response.text.strip():
            st.markdown(response.text)
        else:
            st.warning("Gemini did not return an answer. Please try again")
            st.stop()

            
    except Exception as e:
        st.error("Unable to generate an answer.Please try again.")
        print(f"Gemini API error: {type(e).__name__}: {e} ")
        st.stop()
        
        
    
    with st.expander("📚 View Retrieved PDF Sources"):
        
        for i , position in enumerate (reranked_indices[:3] , start =1):
            chunk_id = sorted_rrf[int(position)][0]
            st.markdown(f"### 📄 Source {i} | Chunk ID: {chunk_id}")
            st.write(chunks[chunk_id])
            st.divider()
            
            
            
        
        
    
    
    

    
   
    
    