import json
import os
from google import genai


api_key = os.getenv("GEMINI_API_KEY")

if not api_key :
    raise SystemExit(" Gemini API Key is Missing. please export it first")

client = genai.Client(api_key=api_key)



with open("rag_evaluation_results.json", "r", encoding="utf-8") as file:
    saved_results = json.load(file)
  

judge_prompts = []
for i in range(len(saved_results)) :

    question = saved_results[i]["question"]
    answer = saved_results[i]["generated_answer"]
    context = saved_results[i]["retrieved_context"]
    
    prompt = f"""
        You are a RAG faithfulness evaluator.

        Question:
        {question}

        Retrieved PDF Context:
        {context}

        Generated Answer:
        {answer}

        Check whether every factual claim in the generated answer
        is supported by the retrieved PDF context.

        Return only:
        1 - If every factual claim is supported.
        0 - If any factual claim is unsupported or contradicted.
    
    """
    
    judge_prompts.append(prompt)
    print("Judge promt prepare for the Question :", i+1)
  
judge_scores = []

for i , prompt in  enumerate(judge_prompts) :
    
    judge_response = client.models.generate_content(
        model = "gemini-3-flash-preview" ,
        contents = prompt
    )    
     
    try:
        judge_score = int(judge_response.text.strip())

        if judge_score in [0, 1]:
            judge_scores.append(judge_score)
            print("Question", i + 1, "Faithfulness Score:", judge_score)
        else:
            print("Invalid score for question:", i + 1)

    except (ValueError, AttributeError): 
        print("Invalid Gemini response for question:", i + 1)
    
        

print("All Judge scores:", judge_scores)

print("Total questions:", len(saved_results))
print("Valid scores received:", len(judge_scores))


if len(judge_scores) == len(saved_results) and len(saved_results) > 0:
    faithfulness = (sum(judge_scores) / len(judge_scores)) * 100
    print(f"LLM Judge Faithfulness: {faithfulness:.2f}%")
else:
    print("Evaluation incomplete. Cannot calculate faithfulness.")
    

with open("rag_judge_scores.json", "w", encoding="utf-8") as file:
    json.dump(judge_scores, file, indent=4)

print("Judge scores saved successfully!")






print("\n Faithfullness testing")
raise SystemExit
    
    
    

    

# print("Total evaluation questions:", len(saved_results))
# print("First question:", saved_results[0]["question"])
# print("First Gemini answer:", saved_results[0]["generated_answer"])
# print("First PDF context:", saved_results[0]["retrieved_context"])

##======= MANUVAL EVALUATION ===========##

# Evaluation 1 

answer = saved_results[0]["generated_answer"]
context = saved_results[0]["retrieved_context"]

print("Adam mentioned in the answer:", "adam" in answer.lower())
print("Adam mentioned in PDF context:", "adam" in context.lower())

print("Adan found in both:", "adam" in answer.lower() and "adam" in context.lower())

print("Position of Adam in context:", context.lower().find("adam"))

position = context.lower().find("adam")

print("Evidence:", context[max(0,position-100):position+200] if position !=0 else "Adam not found")

#faithfulness_scores = [1,1,1]


# Evaluation 2 

answer_2 = saved_results[1]["generated_answer"]
context_2 = saved_results[1]["retrieved_context"]

print("Question2 Gemini answer :", answer_2)

print("Encoder menstioned :", "encoder" in context_2.lower())
print("Decoder menstioned :", "decoder" in context_2.lower())

print("\n=========Question2 PDF Evidence=======")
print(context_2)

# Evaluation 3

answer_3 = saved_results[2]["generated_answer"]
context_3 = saved_results[2]["retrieved_context"]

print("\n Question 3 answer is:", answer_3)

print("\n=========Question3 PDF Evidence=======")
print(context_3)


faithfulness_scores = [1, 1, 1]

if len(faithfulness_scores) > 0: 
    faithfulness_percentage = (
        sum(faithfulness_scores) / len(faithfulness_scores)
    ) * 100
else :
    faithfulness_scores = 0 


if len(faithfulness_scores) == 0 :
    print("No answers evaluated yet.") 
else :
    print ("RAG Faithfulness:", faithfulness_percentage,"%")
    


#======= Automatic LLM FAITHFULNESS Evaulations ========= ##

judge_answer = saved_results[0]["generated_answer"]
judge_context = saved_results[0]["retrieved_context"]

judge_prompt = f"""

You are a RAG faithfulness evaluator 

Retrieved PDF Context :
{judge_context}

Generated_answer :
{judge_answer}

check whether every factual claim in the generated answer 
is supported by the retrieved PDF contex.

Return only :

1 - If every factual claim is supported
0 - If any factual claim is unsupported or contradicted.

"""
 
print(judge_prompt)
print("The length of the prompt:", len(judge_prompt))

# judge_response = client.models.generate_content(
#     model = "gemini-3-flash-preview",
#     contents= judge_prompt
# )

# print("LLM Judge Faithfulness score:", judge_response.text)


# try:
#     judge_score = int(judge_response.text.strip())

#     if judge_score in [0, 1]:
#         print("Valid LLM Judge score:", judge_score)
#     else:
#         print("Invalid LLM Judge score!")

# except (ValueError, AttributeError):
#     print("Gemini did not return a valid numerical score.")






