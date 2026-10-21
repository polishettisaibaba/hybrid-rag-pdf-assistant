test_response = "1"

try:
    test_score = int(test_response)

    if test_score in [0, 1]:
        print("Valid faithfulness score:", test_score)
    else:
        print("Invalid score! Only 0 or 1 allowed.")

except ValueError:
    print("Conversion failed! Expected a number.")
    

for i in range(3):

    print("Evaluating question :", i+1)
    
    
questions = ["Adam optimizer", "Encoder layers", "Multi-head attention"]

for i in range(len(questions)):
    judge_prompt = f"Evaluate the answer about {questions[i]}"
    print(judge_prompt)


