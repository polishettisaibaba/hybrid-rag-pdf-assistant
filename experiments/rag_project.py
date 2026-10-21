from pypdf import PdfReader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rank_bm25 import BM25Okapi 
from sentence_transformers import CrossEncoder
import json

pdf_path = "attention_is_all_you_need.pdf"
reader = PdfReader(pdf_path)

print("Numeber of pages: ", len(reader.pages))

#Extract actual text from the PDF : ( checked by doing for one page)

first_page = reader.pages[0]

text = first_page.extract_text()
#print(text)

## Extracting full pdf

all_text = ""

for page in reader.pages :
    page_text = page.extract_text()
    all_text = all_text + page_text + "\n"
    
print("Total characters:", len(all_text))

# CHUNKING 

# chunk_size = 1000 

# chunks = []

# for i in range(0,len(all_text),chunk_size):
#     chunk = all_text[i:i+chunk_size]
#     chunks.append(chunk)
    
# print("Number of chunks: ", len(chunks))

# print("\n First chunk is :")
# print(chunks[0])
# print("\n Second chunk is :",)
# print(chunks[1])


## Added chunk overlap 
chunk_size = 1000
overlap = 200 

chunks_overlap = []

step_size = chunk_size - overlap

for i in range(0,len(all_text),step_size):       # step size changed unlike before chunk size
    chunk = all_text[i:i+chunk_size]           # We didnot change chunk size-observe
    chunks_overlap.append(chunk)
    
print("Number of chunks with overlap: ", len(chunks_overlap))

print("\n End of chunk is :")
print(chunks_overlap[0][-200:])
print("\n Start of chunk 2 is :",)
print(chunks_overlap[1][:200])

# SMARTER CHUNKING 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators = ["\n\n","\n"," ",""]
)

smart_chunks = text_splitter.split_text(all_text)      # we shall use this next while embedding 

### Metadata -aware chunks for source tracking

chunks_with_metadata = [] 

for page_number, page in enumerate(reader.pages,start=1):
    
    metadata_page_text = page.extract_text()
    page_chunks = text_splitter.split_text(metadata_page_text)
    
    for page_chunk in page_chunks :
        
        chunk_data = {
            "chunk_id": len(chunks_with_metadata),
            "text" : page_chunk,
            "page" : page_number,
            "source": pdf_path
        }

        chunks_with_metadata.append(chunk_data)
        
        
print("\n Number of metadata chunks:", len(chunks_with_metadata))

print("\n First Metadata chunk: ")
print(chunks_with_metadata[0])


# Embedding Model

model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract text only for creating chunks

metadata_texts = []

for chunk in chunks_with_metadata :
    metadata_texts.append(chunk["text"])
    
print("\n Number of Metadata texts:", len(metadata_texts))

print("\n First metadata text: ")
print(metadata_texts[0])

# BM25 KEYWORD RETRIEVAL

tokenized_corpus = []

for text in metadata_texts:
    tokens = text.lower().split()
    tokenized_corpus.append(tokens)

bm25 = BM25Okapi(tokenized_corpus)

#Creating Embeddings for the metadata-aware chunks

metadata_embeddings  = model.encode(metadata_texts)
print("\n Metadata embeddings shape :", metadata_embeddings.shape)

#Prepared these embeddings for FAISS

metadata_embeddings_faiss = metadata_embeddings.astype("float32").copy()
faiss.normalize_L2(metadata_embeddings_faiss)   # Normalise the whole embedding

#CREATED the New FAISS index 

metadata_dimension = metadata_embeddings_faiss.shape[1]  # Get the embedding dimension
metadata_index = faiss.IndexFlatIP(metadata_dimension)  # Created empty faiss index by taking dimension consideration
metadata_index.add(metadata_embeddings_faiss) # Add (all 52) Prepared embeddings for FAISS in to the created empty faiss index
# print("\n Number of vectors in metadata FAISS Index :",metadata_index.ntotal) #Verify that all 52 embedding vectors were added to the FAISS index-
#( i,e ntotal is total no of vectors currently stored in FAISS)

#TEST QUESTION FOR METADATA-AWARE RETRIEVEL

#metadata_question = "What BLEU score did the Transformer achieve?"   # Main question 
#metadata_question = "Who is the CEO of Microsoft?"   #Test question irrelevant one
# Test 1 — Relevant
metadata_question = "What optimizer was used to train the Transformer?" 
#Test
#metadata_question = "How many encoder and decoder layers are used in the Transformer?"
## metadata_question = "How many encoder and decoder layers are used in the Transformer?"
#Test
#metadata_question = "Who is the CEO of Microsoft?"
#metadata_question = "What is multi-head attention?"



tokenized_query = metadata_question.lower().split()

print("\nBM25 Query Tokens:")
print(tokenized_query)
bm25_scores = bm25.get_scores(tokenized_query)
print("\nNumber of BM25 scores:", len(bm25_scores))

bm25_ranked_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)

print("\n===== BM25 RANKING =====")

for rank, index_id in enumerate(
    bm25_ranked_indices[:15],
    start=1
):
    print(
        "Rank:", rank,
        "| Chunk ID:", index_id,
        "| Score:", round(float(bm25_scores[index_id]), 4)
    )
    



# Test 2 — Relevant
# metadata_question = "How many encoder and decoder layers are used in the Transformer?"
# Test 3 — Irrelevant
# metadata_question = "Who is the CEO of Microsoft?"
# Test 4 — Relevant
# metadata_question = "What is multi-head attention?"


metadata_question_embedding = model.encode([metadata_question]).astype("float32") #ENCODING THE Question with same model

faiss.normalize_L2(metadata_question_embedding) #Normalise the question as well

# SEARCH METADATA FAISS          

metadata_top_k = 30

metadata_scores, metadata_indices = metadata_index.search(
    metadata_question_embedding,
    metadata_top_k
)

print("\n Metadata FAISS indices:", metadata_indices)
print("Meta FAISS scores :", metadata_scores)


print("\n===== METADATA FAISS RANKING =====")

for rank, (index_id, score) in enumerate(
    zip(metadata_indices[0], metadata_scores[0]),
    start=1
):
    print(
        "Rank:", rank,
        "| Chunk ID:", index_id,
        "| Score:", round(float(score), 4)
    )


print("\n===== FINDING ADAM CHUNK =====")

for item in chunks_with_metadata:
    if "Adam optimizer" in item["text"]:
        print(
            "Chunk ID:", item["chunk_id"],
            "| Page:", item["page"]
        )
        print(item["text"])



print("\n===== FINDING EXACT RANK OF ADAM CHUNK =====")

all_scores, all_indices = metadata_index.search(
    metadata_question_embedding,
    len(chunks_with_metadata)
)

for rank, (index_id, score) in enumerate(
    zip(all_indices[0], all_scores[0]),
    start=1
):
    if "Adam optimizer" in chunks_with_metadata[index_id]["text"]:
        print(
            "Rank:", rank,
            "| Chunk ID:", index_id,
            "| Score:", round(float(score), 4)
        )
        break


print("\n===== KEYWORD SEARCH: OPTIMIZER =====")

for item in chunks_with_metadata:
    if "optimizer" in item["text"].lower():
        print(
            "Chunk ID:", item["chunk_id"],
            "| Page:", item["page"]
        )




# HYBRID RETRIEVAL USING RECIPROCAL RANK FUSION

rrf_scores = {}
rrf_k = 60

# Add FAISS ranks to RRF scores

for rank, index_id in enumerate(metadata_indices[0], start=1):
    rrf_scores[int(index_id)] = 1 / (rrf_k + rank)


# Add BM25 ranks to the same RRF scores

for rank, index_id in enumerate(bm25_ranked_indices[:15], start=1):
    index_id = int(index_id)

    if index_id not in rrf_scores:
        rrf_scores[index_id] = 0

    rrf_scores[index_id] += 1 / (rrf_k + rank)


# SORT THE HYBRID RRF Scores :

hybrid_ranked = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse = True
)

print("\n ==== HYBRID RRF RANKING=====")

for rank , (chunk_id,score) in enumerate(
    hybrid_ranked[:15],
    start =1
):
    print(
        "Rank : ", rank,
        "| ChunK ID", chunk_id,
        "| RRF Score", round(score,5)
    )


# STEP 1: LOAD CROSS-ENCODER RERANKING MODEL

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("\nCross-Encoder reranker loaded successfully!")


# STEP 2: PREPARE QUESTION-CHUNK PAIRS FOR RERANKING

rerank_pairs = []

for chunk_id, rrf_score in hybrid_ranked[:15]:

    chunk_text = chunks_with_metadata[chunk_id]["text"]

    pair = (metadata_question, chunk_text)

    rerank_pairs.append(pair)

print("\nNumber of Cross-Encoder pairs:", len(rerank_pairs))
print("\nFirst question:", rerank_pairs[0][0])
print("\nFirst chunk:", rerank_pairs[0][1][:200])

# STEP 3: GENERATE CROSS-ENCODER RELEVANCE SCORES

rerank_scores = reranker.predict(rerank_pairs)

print("\nNumber of Cross-Encoder scores:", len(rerank_scores))


# STEP 4: First 5 CROSS ENCODER SCORES FOR THE QUESTION CHUNK PAIR.

print("\n FirST 5 Cross -Encoderw scores : ")
print(rerank_scores[:5])

# STEP 5: CONNECT CHUNK IDS WITH CROSS-ENCODER SCORES 

reranked_results = []

for i , score in enumerate (rerank_scores):
    chunk_id = hybrid_ranked[i][0]
    reranked_results.append((chunk_id,float(score)))
    
print("\n First 5 chunk IDs with Cross - Encoders scores")
print(reranked_results[:5])

# STEP 6: SORT CHUNKS BY CROSS-ENCODER RELEVANCE
    
reranked_results = sorted(
    
    reranked_results,
    key = lambda item : item[1],
    reverse = True
)
print("\nTop 5 Cross-Encoder reranked chunks:")
print(reranked_results[:5])    
    
# STEP 7: SELECT TOP  CROSS-ENCODER CHUNKS as Per K requirement 

reranked_top_k = 5

top_reranked_chunks = reranked_results[:reranked_top_k]

print(f"\nTop {reranked_top_k} Cross-Encoder selected chunks:")

for chunk_id, score in top_reranked_chunks:
    print(f"Chunk ID: {chunk_id} | Score: {score:.4f}")   

# STEP 8: BUILD CONTEXT FROM CROSS-ENCODER RESULTS

reranked_context = ""

for chunk_id ,score in top_reranked_chunks :
    chunk = chunks_with_metadata[chunk_id]
    
    reranked_context += (
        f"Source : {chunk["source"]} | page : {chunk["page"]} \n "
        f"{chunk["text"]}\n\n"
    )

print("\nCross-Encoder context length:", len(reranked_context))
    








# SELECT TOP 3 HYBRID CHUNKS

hybrid_top_k = 3

hybrid_top_chunks = []

for chunk_id, score in hybrid_ranked[:hybrid_top_k]:
    chunk_data = chunks_with_metadata[chunk_id]
    hybrid_top_chunks.append(chunk_data)

print("\n===== TOP 3 HYBRID CHUNKS =====")

for chunk in hybrid_top_chunks:
    print(
        "Chunk ID:", chunk["chunk_id"],
        "| Page:", chunk["page"]
    )


# BUILD CONTEXT FOR TOP 3 HYBRID CHUNKS 

hybrid_context_parts = []

for chunk in hybrid_top_chunks :
    
    context_block = f'''
    
    source : {chunk["source"]}
    page :  {chunk["page"]}
    chunk ID :  {chunk["chunk_id"]}
    Text :
    {chunk["text"]}
    
    '''
    
    hybrid_context_parts.append(context_block)
    
hybrid_context = "\n\n".join(hybrid_context_parts)

print("\n Hybrid context length :", len(hybrid_context))


print("\n===== HYBRID TOP 3 RRF SCORES =====")

for chunk_id, score in hybrid_ranked[:3]:
    print(
        "Chunk ID:", chunk_id,
        "| RRF Score:", round(score, 5)
    )



 
 # CREATE PROMPT FOR HYBRID RAG

reranked_prompt = f"""
Answer the question using ONLY the context provided below.

Give a short and direct answer.

Mention the source document and page number.

If the answer is not available in the context, say:
"I cannot provide the answer in the provided context."

CONTEXT:
{reranked_context}

QUESTION:
{metadata_question}

ANSWER:
"""


# INITIALIZE GEMINI CLIENT

from google import genai
import os

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not available.")

client = genai.Client(api_key=api_key)
 
 
 
 
# GENERATE ANSWER USING HYBRID RAG CONTEXT

# reranked_response = client.models.generate_content(
#     model="gemini-3-flash-preview",
#     contents= reranked_prompt
# )

# reranked_answer = reranked_response.text

# print("\n===== ReRANKED RAG GEMINI ANSWER =====")
# print(reranked_answer)


# STEP 10: CREATE RAG EVALUATION DATASET

evaluation_data = [
    {
        "question": "What optimizer was used to train the Transformer?",
        "expected_answer": "Adam optimizer",
        "relevant_chunk_ids": [54]
    },
    {
      "question": "How many encoder and decoder layers are used in the Transformer?",
        "expected_answer": "6 encoder layers and 6 decoder layers",
        "relevant_chunk_ids": [18, 20]  
    },
    {
        "question": "What is multi-head attention?",
        "expected_answer": "Multiple attention heads attend to different representation subspaces at different positions.",
        "relevant_chunk_ids": [30]
    }
     
]

print("\nNumber of evaluation questions:", len(evaluation_data))
print("First evaluation question:", evaluation_data[0]["question"])


# STEP 11: CALCULATE CROSS-ENCODER HIT@K

relevant_ids = evaluation_data[0]["relevant_chunk_ids"]

retrieved_ids = [chunk_id for chunk_id, score in top_reranked_chunks]

hit = any(chunk_id in relevant_ids for chunk_id in retrieved_ids)

print("\nRelevant chunk IDs:", relevant_ids)
print("Retrieved chunk IDs:", retrieved_ids)
print("Cross-Encoder Hit@5:", int(hit))



# COMPARE RRF and CROSS -ENconder Fairly 
# STEP 12: CALCULATE RRF HIT@K

rrf_top_k = 5 

rrf_retrieved_ids = [  
    chunk_id for chunk_id , score in hybrid_ranked[:rrf_top_k]
]

rrf_hit = any(
    chunk_id in relevant_ids for chunk_id in rrf_retrieved_ids
)

print("\nRRF retrieved chunk IDs:", rrf_retrieved_ids)
print(f"RRF Hit@{rrf_top_k}:", int(rrf_hit))

print("\nTotal RRF ranked chunks:", len(hybrid_ranked))
print("RRF Top 5:", hybrid_ranked[:5])

# STEP 13: COMPARE RRF AND CROSS-ENCODER HIT@3

rrf_hit_3 = any(
    chunk_id in relevant_ids
    for chunk_id in rrf_retrieved_ids[:3]
)

reranked_hit_3 = any(
    chunk_id in relevant_ids
    for chunk_id in retrieved_ids[:3]
)

print("\nRRF Hit@3:", int(rrf_hit_3))
print("Cross-Encoder Hit@3:", int(reranked_hit_3))


# STEP 14: CALCULATE RECIPROCAL RANK

def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


rrf_rr = reciprocal_rank(rrf_retrieved_ids, relevant_ids)

reranked_rr = reciprocal_rank(retrieved_ids, relevant_ids)

print("\nRRF Reciprocal Rank:", round(rrf_rr, 4))
print("Cross-Encoder Reciprocal Rank:", round(reranked_rr, 4))

# Preaparing other 2 quetion  for find MRR after caliculating RR for Each 

#Q2: How many encoder and decoder layers are used in the Transformer?
#Q3: What is multi-head attention?

# STEP 15A: FIND CHUNKS FOR NEW EVALUATION QUESTIONS 

# STEP 15F: INSPECT MULTI-HEAD ATTENTION CHUNK

# STEP 15G: FIND MULTI-HEAD ATTENTION DEFINITION

# STEP 16A: LOOP THROUGH ALL EVALUATION QUESTIONS

faiss_rr_scores = [] 
faiss_hit_scores = []

for item in evaluation_data:
    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]
    question_embedding = model.encode([question], normalize_embeddings=True)
    distances , indices = metadata_index.search(question_embedding,5)

    faiss_ids = indices[0].tolist()
    faiss_hit = any(
    chunk_id in relevant_ids
    for chunk_id in faiss_ids
    )
    
    faiss_hit_scores.append(int(faiss_hit))
    
    faiss_rr = reciprocal_rank(faiss_ids, relevant_ids)
    
    faiss_rr_scores.append(faiss_rr)

    
    
    print("\nQuestion:", question)
    print("Relevant Chunk IDs:", relevant_ids)
    print("Embedding shape:", question_embedding.shape)
    print("FAISS retrieved IDs:", indices[0].tolist())
    print("FAISS Hit@5:", int(faiss_hit))
    print("FAISS Reciprocal Rank:", round(faiss_rr, 4))


faiss_mrr = sum(faiss_rr_scores)/len(faiss_rr_scores)
print("\n FAISS MRR@5 :", round(faiss_mrr,4))

faiss_hit_rate = sum(faiss_hit_scores) / len(faiss_hit_scores)
print("FAISS Hit Rate@5:", round(faiss_hit_rate, 4))


# STEP 17A: PREPARE RRF EVALUATION

rrf_rr_scores = []
rrf_hit_scores = []

print("\nRRF evaluation started")

for item in evaluation_data:
    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]
    
    question_embedding = model.encode(
    [question],
    normalize_embeddings=True
    )

    distances, indices = metadata_index.search(
        question_embedding, 30
    )
    faiss_ranked_ids = indices[0].tolist()
    
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    bm25_ranked_ids = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
    )
    
    rrf_scores = {}
    for ranked_ids in [faiss_ranked_ids, bm25_ranked_ids[:15]]:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )
    
    
    rrf_ranked = sorted(
    rrf_scores.items(),
    key=lambda item: item[1],
    reverse=True
    )

    
    rrf_top5_ids = [
    chunk_id for chunk_id, score in rrf_ranked[:5]
    ]
    
    # STEP 17E: CALCULATE RRF HIT@5

    rrf_hit = any(
    chunk_id in relevant_ids
    for chunk_id in rrf_top5_ids
    )
    rrf_hit_scores.append(int(rrf_hit))


    rrf_rr = reciprocal_rank(rrf_top5_ids, relevant_ids)
    rrf_rr_scores.append(rrf_rr)
    
  
    
    
    print("\nRRF Question:", question)
    print("Relevant Chunk IDs:", relevant_ids)
    print("RRF FAISS Top 5 IDs:", indices[0][:5].tolist())
    print("BM25 scores count:", len(bm25_scores))
    print("First 5 BM25 scores:", bm25_scores[:5])
    print("BM25 Top 5 IDs:", bm25_ranked_ids[:5])
    print("Number of RRF Candidates:", len(rrf_scores))
    print("RRF Top 5:", rrf_ranked[:5])
    print("RRF Top 5 Chunk IDs:", rrf_top5_ids)
    print("RRF Hit@5:", int(rrf_hit))
    print("RRF Reciprocal Rank:", round(rrf_rr, 4))


print("\n All RRF Reciprocal Rank scores:", rrf_rr_scores)

# RRF MRR@5 

rrf_mrr = sum(rrf_rr_scores) / len(rrf_rr_scores)
print("\n RRF MRR@5:", round(rrf_mrr, 4))

rrf_hit_rate = sum(rrf_hit_scores) / len(rrf_hit_scores)
print("RRF Hit Rate@5:", round(rrf_hit_rate, 4))


# STEP 18A: PREPARE CROSS-ENCODER EVALUATION

reranker_rr_scores = []
reranker_hit_scores = []
ce_rr_scores = []
ce_hit_scores = []
evaluation_retrieved_ids = []

print("\n Cross-ENCODER EVALUATION STARTED")

# STEP 18B: LOOP THROUGH EVALUATION QUESTIONS

for item in evaluation_data:

    question = item["question"]
    relevant_ids = item["relevant_chunk_ids"]
    
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )
    
    # STEP 18D: RETRIEVE FAISS CANDIDATES

    distances, indices = metadata_index.search(
        question_embedding, 30
    )

    faiss_ranked_ids = indices[0].tolist()

    tokenized_query = question.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_ranked_ids = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )

    # STEP 18F: COMBINE RANKINGS USING RRF

    rrf_scores = {}

    for ranked_ids in [faiss_ranked_ids, bm25_ranked_ids[:15]]:

        for rank, chunk_id in enumerate(ranked_ids, start=1):

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0)
                + 1 / (60 + rank)
            )

    
    # STEP 18G: SORT RRF CANDIDATES

    rrf_ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )
    rrf_top15 = rrf_ranked[:15]

    # STEP 18H: PREPARE CROSS-ENCODER PAIRS

    reranker_pairs = [
        (question, chunks_with_metadata[chunk_id]["text"])
        for chunk_id, score in rrf_top15
    ]

    # STEP 18I: CALCULATE CROSS-ENCODER SCORES

    ce_scores = reranker.predict(reranker_pairs)

   # STEP 18J: SORT CROSS-ENCODER RESULTS

    cross_encoder_ranked = sorted(
        zip(rrf_top15, ce_scores),
            key=lambda item: item[1],
            reverse=True
    )
   
    # STEP 18K: DISPLAY TOP 5 CROSS-ENCODER RESULTS
    cross_encoder_top5 = cross_encoder_ranked[:5]
    # STEP 18L: EXTRACT TOP 5 CHUNK IDS

    ce_top5_ids = [
        chunk_id
             for (chunk_id, rrf_score), ce_score in cross_encoder_top5
    ]

    evaluation_retrieved_ids.append(ce_top5_ids.copy())
    
    
    
    # STEP 18M: CALCULATE CROSS-ENCODER HIT@5

    ce_hit = any(
        chunk_id in relevant_ids
        for chunk_id in ce_top5_ids
    )
    ce_hit_scores.append(int(ce_hit))
    

    print("\nCross-Encoder Top 5:")

    for (chunk_id, rrf_score), ce_score in cross_encoder_top5:
        print(
        "Chunk ID:", chunk_id,
        "| Cross-Encoder Score:", round(float(ce_score), 4)
        )

    # STEP 18N: CALCULATE CROSS-ENCODER RECIPROCAL RANK

    ce_rr = reciprocal_rank(ce_top5_ids, relevant_ids)
    ce_rr_scores.append(ce_rr)
       


    print("\nCross-Encoder Question:", question)
    print("Relevant Chunk IDs:", relevant_ids)
    print("Question embedding shape:", question_embedding.shape)
    print("FAISS candidates count:", len(faiss_ranked_ids))
    print("FAISS Top 5 IDs:", faiss_ranked_ids[:5])
    print("BM25 Top 5 IDs:", bm25_ranked_ids[:5])
    print("Number of RRF candidates:", len(rrf_scores))
    print("RRF Top 15 IDs:", [chunk_id for chunk_id, score in rrf_top15])
    print("Number of Cross-Encoder pairs:", len(reranker_pairs))
    print("Number of Cross-Encoder scores:", len(ce_scores))
    print("Cross-Encoder Top 5 IDs:", ce_top5_ids)
    print("Cross-Encoder Hit@5:", int(ce_hit)) 
    print("Cross-Encoder Reciprocal Rank:", round(ce_rr, 4)) 



ce_mrr = sum(ce_rr_scores) / len(ce_rr_scores)
print("\n Cross-Encoder MRR@5:", round(ce_mrr, 4))


ce_hit_rate = sum(ce_hit_scores) / len(ce_hit_scores)

print("\nRetrieved chunks for all questions:")
print(evaluation_retrieved_ids)





print("\n===== RAG RETRIEVAL EVALUATION =====")

print("RRF Hit Rate@5:", round(rrf_hit_rate, 4))
print("Cross-Encoder Hit Rate@5:", round(ce_hit_rate, 4))

print("RRF MRR@5:", round(rrf_mrr, 4))
print("Cross-Encoder MRR@5:", round(ce_mrr, 4))



# STEP 20B: PREPARE ANSWER QUALITY EVALUATION

print("\n===== ANSWER QUALITY EVALUATION =====")

generated_answers = []
correctness_scores = []
faithfulness_scores = []
evaluation_contexts = []

manual_scores = [1,1,1]

question_index = 0

for item in evaluation_data:
    question = item["question"]
    expected_answer = item["expected_answer"]
    retrieved_ids = evaluation_retrieved_ids[question_index]
    
    retrieved_texts = [
        chunks_with_metadata[chunk_id]["text"]
        for chunk_id in retrieved_ids
    ]
    
    context = "\n\n".join(retrieved_texts)
    evaluation_contexts.append(context)
    
    evaluation_prompt = f"""
    
    you are a research assiastant answering questions about a PDF document
    
    Use only the provided context to answer the question.
    IF the answer is not avilable in the context , say :
    "I cannot find the answer in the provided document."
    
    context : 
    {context}
    
    Question :
    {question}
    
    Answer : 
    
    """
    
    
    evaluation_response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=evaluation_prompt
    )

    generated_answer = evaluation_response.text

    generated_answers.append(generated_answer)
    correctness_scores.append(manual_scores[question_index])
    
    print("\n-------GEMENI ANSWER-----")
    print("Generated answer:", generated_answer)    
    
    
    
    
    print("\n" + "="*60)
    print("Question:", question_index +1)
    print("\n"+ "="*60)
    
    print("\nQuestion:", question)
    print("Expected Answer:", expected_answer)  
    
    print("\n ----RETRIEVAL DETAILS------")
    print("\nRetrieved Chunk IDs:", retrieved_ids)
    print("Number of Retrived chunks:", len(retrieved_texts))
    print("Combined context length:", len(context), "characters")
    
    print("\n-----PDF CONTEXT PREVIEW--------")
    print(context[:300],"....")
    
    question_index += 1
    

evaluation_results = []

for i in range(len(evaluation_data)):
    evaluation_results.append({
        "question": evaluation_data[i]["question"],
        "generated_answer": generated_answers[i],
        "retrieved_context": evaluation_contexts[i]
    })

with open("rag_evaluation_results.json", "w", encoding="utf-8") as file:
    json.dump(evaluation_results, file, indent=4)


print("Evaluation results saved successfully")


    
 
# STEP 20L.6: CALCULATE MANUAL ANSWER ACCURACY

if len(correctness_scores) > 0:
    answer_accuracy = sum(correctness_scores) / len(correctness_scores)
else:
    answer_accuracy = 0

print("\n===== MANUAL ANSWER EVALUATION =====")
print("Answer Accuracy:", round(answer_accuracy * 100, 2), "%") 
    

print("\n===== FAITHFULNESS PREPARATION =====")

print("Total generated answers:", len(generated_answers))
print("Total evaluation contexts:", len(evaluation_contexts))

if len(generated_answers) == len(evaluation_contexts):
    print("Answer-context counts match!")
else:
    print("WARNING: Answer-context counts do not match!")


print("\n===== ANSWER-CONTEXT PAIRS =====")

for answer, context in zip(generated_answers, evaluation_contexts):
    print("Answer:", answer[:100])
    print("Context Preview:", context[:300])
    print("Full retrieved context:\n", context)
    print("Full context avilable:", len(context)>0)
    print("Context length:", len(context))
    print("-" * 40)



print("\n===== FAITHFULNESS REVIEW =====")

for question_number, (answer, context) in enumerate(
    zip(generated_answers, evaluation_contexts), start=1
):
    print("\nQuestion:", question_number)
    print("Gemini Answer:", answer)
    print("\nSupporting PDF Context:")
    print(context)
    print("=" * 60)







# TEMPORARY STOP - TEST Re ranked RANKING ONLY
print("\n Reranked ranking test completed.")
raise SystemExit








#FILTER METADATA RESULTS USING SIMILARITY THRESHOLD. 

similarity_threshold = 0.30 
retrieved_metadata=[]

for index_id , score in zip(metadata_indices[0],metadata_scores[0]):

    if score >= similarity_threshold:
        chunk_data = chunks_with_metadata[index_id]
        
        retrieved_metadata.append({
          "chunk_id": chunk_data["chunk_id"],
            "text": chunk_data["text"],
            "page": chunk_data["page"],
            "source": chunk_data["source"],
            "score": float(score)  
            
        })

print("\n Number of relevant metadata chunks: ", len(retrieved_metadata))

if len(retrieved_metadata) == 0 :
    print("\n No Sufficiently relevant information was found in the documents." )
    exit()
    
    
## BUILD CONTEXT from relevant metadata chunks

metadata_context_parts = []

for item in retrieved_metadata:
    metadata_context_parts.append(item["text"])
    
metadata_context  = "\n\n".join(metadata_context_parts)
print("\n Metadata context length :", len(metadata_context))


## BUILD citation - aware context 

citation_context_parts = []

for item in retrieved_metadata : 
    
    citation_block = f'''
        source : {item["source"]}
        Page: {item["page"]}
        Chunk ID: {item["chunk_id"]}   
        text: 
        {item["text"]}

         '''

    citation_context_parts.append(citation_block)
    
citation_context = "\n\n".join(citation_context_parts)
print(citation_context)

    
    
    

## RETRIEVE METADATA FOR THOSE FAISS RESULTS.

for rank, (index_id, score) in enumerate(
    zip(metadata_indices[0], metadata_scores[0]),
    start=4
):
    
    chunk_data = chunks_with_metadata[index_id]

    print("\n--------------------")
    print("Rank:", rank)
    print("Chunk ID:", chunk_data["chunk_id"])
    print("Page:", chunk_data["page"])
    print("Source:", chunk_data["source"])
    print("Similarity score:", round(float(score), 4))
    print("Text:")
    print(chunk_data["text"])







## above is for metadata- aware

## below is the contiue of smartchunks sections
# print("Number of smart chunks:", len(smart_chunks))

# print("\n Smart chunk 1 :")
# print(smart_chunks[0])

# print("\n Smart chunk 2 :")
# print(smart_chunks[1])

# print("\n length of smart chunk 1 :", len(smart_chunks[0]))
# print("\n length of smart chunk 2 :", len(smart_chunks[1]))  # Found length is not 1000 at all chunks

## EMBEDDINGS

# sentence = "The Transformer uses attention mechanisms."

# embedding = model.encode(sentence)

# print("\n Embedding:")
# print(embedding)

# print("\n Embedding dimension:", len(embedding))

## Experiment of Semantic similarity. 

# sentence_1 = "The cat is sitting on the sofa"
# sentence_2 = "A kitten is resting on the counch"
# sentence_3 = "Python is a programming language"

# emb1 = model.encode(sentence_1)
# emb2 = model.encode(sentence_2)
# emb3 = model.encode(sentence_3)

# similarity_1_2 = cosine_similarity([emb1],[emb2])[0][0]
# similarity_1_3 = cosine_similarity([emb1],[emb3])[0][0]

# print(" cat vs kitten similarity :", similarity_1_2)
# print(" cat vs python similarity :", similarity_1_3)

### Now did EMBEDING TO OUR ACTUALLY PDF :- ( we used smart chunks we did before)


#chunk_embeddings = model.encode(smart_chunks)
    
# print("\n Number of chunk embeddings: ", len(chunk_embeddings))
# print("\n Embeding dimensions of each chunk :", len(chunk_embeddings[0]))

# print(" Shape of chunk embedings :", chunk_embeddings.shape) 

## Above - each chunk dimensions are typically equal to features as similar concept of ML. 
## Above shape of whole pdf embedings we have (50, 384)-that is 50 chunks and 384 dimesion to eachchunk i.e final_matric= (50,384) 

## RETRIVEL WITH MANUAL COSINE SEMILIRATIY
# FIRST REAL RETRIEVAL DONE BELOW: ( which finds evidence that can help answer the question i.e gives required context.) 

question = "What BLEU score did the Transformer achieve?"
question_embeding = model.encode(question)

similarities = cosine_similarity(
    [question_embeding],
    chunk_embeddings
)[0]

print(" \n Number of smilarity scores: ", len(similarities))

best_chunk_index = similarities.argmax()

best_score = similarities[best_chunk_index]
best_chunk = smart_chunks[best_chunk_index]

# print("\n Question :")
# print(question)

# print("\n Best matching chunk index")
# print(best_chunk_index)

# print("\n Similarity score of the question asked in the emddings of whole pdf")
# print(best_score)

# print("\n Retrived Chunk data from the pdf ")
# print(best_chunk)

# Above we have done top 1 retrivel 

## Below we did TOP-K Retrieval 

top_k = 3

top_indices = similarities.argsort()[-top_k:][::-1]

print("\n Top three chunk indices: ", top_indices)

print(" \n ==== TOP THREE RETRIVED CHUNKS =====") 

for rank, index in enumerate(top_indices,start=1):
    
    print("\n------------------")
    print("Rank:", rank)
    print("chunk index :", index)
    print("similarity score:", similarities[index])
    print("\n--------------")
    
    print(smart_chunks[index])
    

#=======================
#CREATE FAISS VECTOR INDEX 
#=======================

dimension = chunk_embeddings.shape[1]

print("\n Embedding dimensions:", dimension)

##index = faiss.IndexFlatL2(dimension)   ##  L2 mean Euclidean distance -

index = faiss.IndexFlatIP(dimension)    ## IP means Inner Product 

print("Vectors inside the FAISS Before adding:", index.ntotal)

chunk_embeddings_faiss = chunk_embeddings.astype("float32")

faiss.normalize_L2(chunk_embeddings_faiss)

index.add(chunk_embeddings_faiss)

print("Vectors inside the FAISS After adding:", index.ntotal)

##==================================
# SEARCH THE FAISS INDEX
##==================================

#question = "What BLEU score did the Transformer achieve ?" ##1- DID DURING PROJECT CREATION-)

#question = "How many encoder and decoder layers are used in the Transformer?" #(-test1 after completion of project
#question = "Who is the CEO of Microsoft?"  ## -Test2 - we are asking question which our pdf dont have answer-hallucination testing and groundedness test
#question = "What optimizer was used to train the Transformer?" #Test 3
#question = "What is multi-head attention?"  # Test 4
#question = "How long did it take to train the big Transformer model?" #Test 5
#question = "What is the capital of France?" # Test 6
#question = "Who invented the telephone?" # Test 7

question_embedding_faiss = model.encode([question]).astype("float32")

faiss.normalize_L2(question_embedding_faiss)

top_k = 3 

scores, indices = index.search(
    question_embedding_faiss,
    top_k
)

print("\n FAISS Top-K indices :", indices)
print("FAISS similarity scores:", scores)

#####################################################
# COMPARISION COSINE SIMILARITY TOP-K (VS) FAISS TOP-K 
######################################################

print("\n Top three chunk indices: ", top_indices)
print("\n FAISS Top-K indices :", indices)

#===============================
#CONVERT FAISS INDICES back to TEXT 
#=================================

retrieved_chunks = []
retrieved_scores = []

# for index_id in indices[0]:                               ## These we used before evaluation -
#     retrieved_chunks.append(smart_chunks[index_id])

similarity_threshold  = 0.30               ## Added this below, after evaluation of similarity score- trying to keep only relavent chunks
   
for index_id , score in zip(indices[0], scores[0]):
    if score >= similarity_threshold:
        retrieved_chunks.append(smart_chunks[index_id])
        retrieved_scores.append(float(score))
        
        

if len(retrieved_chunks)== 0 :
    print("\n No sufficciently relevant information was found in the documents.")
    exit()

    
print("\n==THE RETRIVED CHUNKS FROM THE FAISS ====")

for rank,chunk in enumerate(retrieved_chunks,start=1):
    
    # score = scores[0][rank-1]             # Added during evualuation process - so we could display the similarity score besides each retrieved chunk.
    
    score = retrieved_scores[rank-1]
    
    print("\n------------------")
    print("Rank:", rank)
    print ("similarity_score:", round(score),4)
    print("---------------------")
    print(chunk)


# ==========================================
# BUILD CONTEXT FOR THE LLM ( I.e Join all the chunks text you retrived)
# ==========================================

context = "\n\n".join(retrieved_chunks)

print("\n===== CONTEXT FOR LLM =====")
print(context)

print("\nContext length:", len(context))

#above retrival is done. Now we shall do augemented promt.

# ==========================================
# BUILD THE AUGMENTED PROMPT ( which we can directly send to LLM )
# ==========================================

# prompt = f'''


# Answer the question using ONLY the context provided below
# give a short and direct answer

# If the answer is not avaliable in the context , say : "I cannot provide the answer in the provided context."

# CONTEXT :{context}

# QUESTION : {question}

# Answer : 

# '''

# print("\n ====== FINAL RAG PROMPT======")
# print(prompt)

# ==========================================
# METADATA-AWARE RAG PROMPT.  ( Building promt for the metadata_retrived to send for LLM)
# ==========================================

metadata_prompt = f'''

Answer the question using ONLY the context provided below.
Give a short and direct answer.

Use the source and page information provided in the context
to mention where the answer came from.

If the answer is not available in the context, say:
"I cannot provide the answer in the provided context."

CONTEXT:{citation_context}


QUESTION:{metadata_question}


ANSWER:

'''




# # ==========================================
# # - GENERATE ANSWER USING AN LLM. 
# # ==========================================

# llm_name = "google/flan-t5-small"

# tokenizer = AutoTokenizer.from_pretrained(llm_name)
# llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)

# inputs = tokenizer(
#     prompt,
#     return_tensors="pt",
#     truncation=True,
#     max_length = 512
# )

# outputs = llm.generate(
#     **inputs,
#     max_new_tokens=100
# )

# answer = tokenizer.decode(
#     outputs[0],
#     skip_special_tokens=True
# )


# print("\n===== FINAL RAG ANSWER =====")
# print(answer)

# ==========================================
# CONNECTING RAG PROMT TO GEMINI -  
# ==========================================

from google import genai 
import os

#Read GEMINI API key from environment variable 
api_key = os.getenv("GEMINI_API_KEY")

#CREATE Gemini client 

client = genai.Client(api_key = api_key)

# Send our Exsisting RAG prompt 


response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents = metadata_prompt
)

answer = response.text

print ("\n ======= GEMINI RAG ANSWER========")
print (answer)

