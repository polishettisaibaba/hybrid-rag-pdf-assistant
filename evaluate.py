from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
import numpy as np
import faiss 
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder 


## Hit@k checks 

def hit_at_k(retrieved_ids, relevant_ids):
    return int(any(chunk_id in relevant_ids for chunk_id in retrieved_ids))


# Reciprocal ranks checks

def reciprocal_rank( retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids,start =1) :
        if chunk_id in relevant_ids :
            return 1/rank
        
    return 0 

#=====================================

reader = PdfReader("attention_is_all_you_need.pdf")

pdf_text = ""

for page in reader.pages :
    page_text = page.extract_text() or ""
    pdf_text += page_text + "\n"
    
print("PDF Characters:", len(pdf_text))

# Created the same chunks : 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200 
)

chunks = text_splitter.split_text(pdf_text)
print("Total evaluation chunks :",len(chunks))

#Prepared BM25 for evaluation 

tokenized_chunks = [chunk.lower().split() for chunk in chunks]

bm25 = BM25Okapi(tokenized_chunks)
print("BM25 indexed chunks:", len(tokenized_chunks))


#Build FAISS fOR evaluation 

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embedding_model.encode(chunks)

chunk_embeddings = np.asarray(chunk_embeddings,dtype = "float32")

faiss.normalize_L2(chunk_embeddings)
dimension = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(chunk_embeddings)
print("FAISS indexes chunks:", faiss_index.ntotal)


#MRR@5 across Multiple questions

evaluation_questions = [
    {
        "question": "How many encoder and decoder layers are used in the Transformer?",
        "relevant_ids" : [9]
    },
    {
        "question" : "What is positional encoding in the Transformer?",
        "relevant_ids" : [19]
    },
    {
        "question": "What is multi-head attention in the Transformer?",
        "relevant_ids" : [13,14]
    }  
    
]

print("Total evaluation question :", len(evaluation_questions))

#Retrieve Chunks for all three questions :

questions = [item["question"] for item in evaluation_questions]

question_embeddings = embedding_model.encode(questions)

question_embeddings = np.asarray(question_embeddings, dtype="float32")

faiss.normalize_L2(question_embeddings)
scores, indices = faiss_index.search(question_embeddings, 5)

print("FAISS results for all questions:")
print(indices)
print("Results shape:", indices.shape)
print("Scores:", scores)

# create a Function for MRR

rr_scores = [] 

for i , item in enumerate(evaluation_questions):
    retrieved_ids = indices[i].tolist()
    relevant_ids = item["relevant_ids"]
    
    
    rr = reciprocal_rank(retrieved_ids,relevant_ids)
    
    rr_scores.append(rr)
    
    print(f"Question {i+1} Reciprocal Rank : {rr:.4f}")
    
mrr_at_5 = sum(rr_scores)/len(rr_scores)

print("Actual FAISS MRR@5:", round(mrr_at_5,4))

    
# Calculate FAISS Hit@5 across all questions 

hit_scores = []

for i, item in enumerate(evaluation_questions):
    retrieved_ids = indices[i].tolist()
    relevant_ids = item["relevant_ids"]

    hit = hit_at_k(retrieved_ids, relevant_ids)

    hit_scores.append(hit)

    print(f"Question {i + 1} Hit@5: {hit}")

average_hit_at_5 = sum(hit_scores) / len(hit_scores)

print("Actual FAISS Hit@5:", round(average_hit_at_5, 4))


# Evaluating BM25 (semantic search FAISS vs Keyword search BM25)

bm25_results = []

for item in evaluation_questions:
    question = item["question"]

    query_tokens = question.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    top_5_indices = np.argsort(bm25_scores)[::-1][:5]

    bm25_results.append(top_5_indices.tolist())

print("BM25 results for all questions:")

for i, retrieved in enumerate(bm25_results, start=1):
    print(f"Question {i}: {retrieved}")
    
    

bm25_hit_scores = []
bm25_rr_scores = []

for i, item in enumerate(evaluation_questions):
    retrieved_ids = bm25_results[i]
    relevant_ids = item["relevant_ids"]

    hit = hit_at_k(retrieved_ids, relevant_ids)
    rr = reciprocal_rank(retrieved_ids, relevant_ids)

    bm25_hit_scores.append(hit)
    bm25_rr_scores.append(rr)

    print(f"Question {i + 1}: Hit@5={hit}, RR={rr:.4f}")

bm25_hit_at_5 = sum(bm25_hit_scores) / len(bm25_hit_scores)
bm25_mrr_at_5 = sum(bm25_rr_scores) / len(bm25_rr_scores)

print("BM25 Hit@5:", round(bm25_hit_at_5, 4))
print("BM25 MRR@5:", round(bm25_mrr_at_5, 4))
    

#RRF Evaluation :

rrf_k = 60

faiss_top_10_results = []
bm25_top_10_results = []

for i, item in enumerate(evaluation_questions):
    faiss_ids = faiss_index.search(
        question_embeddings[i:i+1], 10
    )[1][0].tolist()

    query_tokens = item["question"].lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_ids = np.argsort(bm25_scores)[::-1][:10].tolist()

    faiss_top_10_results.append(faiss_ids)
    bm25_top_10_results.append(bm25_ids)

print("FAISS Top 10 for Question 1:", faiss_top_10_results[0])
print("BM25 Top 10 for Question 1:", bm25_top_10_results[0])

#RRF score :

rrf_results = []
rrf_top_10_results = []

for faiss_ids, bm25_ids in zip(faiss_top_10_results, bm25_top_10_results):
    rrf_scores = {}

    for rank, chunk_id in enumerate(faiss_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

    for rank, chunk_id in enumerate(bm25_ids, start=1):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (rrf_k + rank)

    sorted_rrf = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    top_5_rrf_ids = [chunk_id for chunk_id, score in sorted_rrf[:5]]
    top_10_rrf_ids = [chunk_id for chunk_id, score in sorted_rrf[:10]]
    

    rrf_results.append(top_5_rrf_ids)
    rrf_top_10_results.append(top_10_rrf_ids)

for i, retrieved in enumerate(rrf_results, start=1):
    print(f"Question {i} RRF Top 5: {retrieved}")
    


# RRF hit@5 and MRR@5

rrf_hit_scores = []
rrf_rr_scores = []

for i, item in enumerate(evaluation_questions):
    retrieved_ids = rrf_results[i]
    relevant_ids = item["relevant_ids"]

    hit = hit_at_k(retrieved_ids, relevant_ids)
    rr = reciprocal_rank(retrieved_ids, relevant_ids)

    rrf_hit_scores.append(hit)
    rrf_rr_scores.append(rr)

    print(f"Question {i + 1}: Hit@5={hit}, RR={rr:.4f}")

rrf_hit_at_5 = sum(rrf_hit_scores) / len(rrf_hit_scores)
rrf_mrr_at_5 = sum(rrf_rr_scores) / len(rrf_rr_scores)

print("RRF Hit@5:", round(rrf_hit_at_5, 4))
print("RRF MRR@5:", round(rrf_mrr_at_5, 4))


# Cross Encoder Evalution :

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Cross-Encoder model loaded successfully!")

ce_results = []
for i , item in enumerate(evaluation_questions) :
    question = item["question"]
    
    candidate_ids = rrf_top_10_results[i] 
    
    pairs = [
        (question,chunks[chunk_id]) for chunk_id in candidate_ids
    ]
    
    print(f"Question {i + 1}: {len(pairs)} Cross- Encoder pairs")
    
    ce_scores = cross_encoder.predict(pairs)
    print(f"Question {i+1} Cross-Encoder scores:", ce_scores)
    
    
    reranked_positions = np.argsort(ce_scores)[::-1]
    
    reranked_chunk_ids = [candidate_ids[pos] for pos in reranked_positions]
    
    print(f"Question {i+1} Cross -Encoder ranked chunks:", reranked_chunk_ids)
    
    ce_results.append(reranked_chunk_ids[:5])
    

#Caliculate Cross-Encoder Hit@5 and MRR@5 

ce_hit_scores = []
ce_rr_scores = []

for i , item in enumerate(evaluation_questions):
    
    retrieved_ids = ce_results[i]
    relevant_ids = item["relevant_ids"]
    
    hit = hit_at_k(retrieved_ids,relevant_ids)
    rr = reciprocal_rank(retrieved_ids,relevant_ids)
    
    ce_hit_scores.append(hit)
    ce_rr_scores.append(rr)
    
    print(f"Question {i+1} : Cross-Encoders HiT@5 = {hit},RR = {rr} ")
    
ce_hit_at_5 = sum(ce_hit_scores)/len(ce_hit_scores)
ce_mrr_at_5 = sum(ce_rr_scores)/len(ce_rr_scores)

print("Cross-Encoder Hit@5:",round(ce_hit_at_5,4))
print("Cross-Encoder MRR@5:", round(ce_mrr_at_5,4))
    
    
    
print("\n=====RAG RETRIEVAL EVALUATION=====")
print(f"{"Method": <18} {"hit@5": <12} {"MRR@5": <12}")
print("_" * 45)

print(f"{'FAISS':<18} {sum(hit_scores)/len(hit_scores):<12.4f} {mrr_at_5:<12.4f}")
print(f"{'BM25':<18} {bm25_hit_at_5:<12.4f} {bm25_mrr_at_5:<12.4f}")
print(f"{'RRF':<18} {rrf_hit_at_5:<12.4f} {rrf_mrr_at_5:<12.4f}")
print(f"{'Cross-Encoder':<18} {ce_hit_at_5:<12.4f} {ce_mrr_at_5:<12.4f}")  
    
    

    
    
    
    



  
      