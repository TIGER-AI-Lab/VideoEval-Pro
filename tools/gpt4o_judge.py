import os
import json
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import argparse

class GPT4oJudge:
    def __init__(self, model_name="gpt-4o-2024-08-06") -> None:
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def __call__(self, question: str, target: str, predicted_answer: str) -> bool:
        prompt = f"""Your job is to look at a question generated from the video, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]. First, I will give examples of each grade, and then you will grade a new example. The following are examples of CORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia Obama and Sasha Obama Predicted answer 1: sashaand maliaobama Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001. ``` These predicted answers are all CORRECT because:-They fully contain the important information in the gold target.-They do not contain any information that contradicts the gold target.-Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.-Hedging and guessing are permissible, provided that the gold target is fully includedand the response contains no incorrect information or contradictions. The following are examples of INCORRECT predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: Malia. Predicted answer 2: Malia, Sasha, and Susan. Predicted answer 3: Barack Obama does not have any children. Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia. Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children. Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer? Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information. ``` These predicted answers are all INCORRECT because:-A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'mnot sure, i think") are also considered incorrect. The following are examples of NOT_ATTEMPTED predicted answers. ``` Question: What is the name of the man's child in the video? Gold target: Malia and Sasha Predicted answer 1: I don't know. Predicted answer 2: I need more context about which Obama you are talking about. Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children. Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one. ``` These predicted answers are all NOT_ATTEMPTED because:-The important information in the gold target is not included in the answer.-No statements in the answer contradict the gold target.

Also note the following things:-For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". -Predicted answers "120k", "124k", and 115k" are all CORRECT. -Predicted answers "100k" and "113k" are INCORRECT. -Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.-The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.-For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.-Do not punish predicted answers if they omit information that would be clearly inferred from the question.-For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".-Consider the question "What award did A pretrainer'sguide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.-For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.-For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.-Do not punish for typos in people's name if it's clearly the same name. -For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "HyoongWon Choong", "HyungwonChung", or "Hyun Won Chung". 

Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer. 
```
Question:{question} 
Goldtarget:{target} 
Predictedanswer:{predicted_answer} 
``` 
Grade the predicted answer ofthe question as one of: A: CORRECT B: INCORRECT C: NOT_ATTEMPTED Just return the letter "A", "B", or "C", with no text around it.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=5
        )

        answer = response.choices[0].message.content.strip()
        return answer[0].upper() == "A"

def option_judge(gt, response):
    response = response.lower()
    if "the answer is" in response:
        response = response.split("the answer is")[-1].strip()
    elif "answer:" in response:
        response = response.split("answer:")[-1].strip()
    elif "the option is" in response:
        response = response.split("the option is")[-1].strip()
    for char in response:
        if char.isalpha():
            response = char
            break
    return response[0] == gt or response[0] == gt.lower() if len(response) > 0 else False

def safe_judge_with_retry(model, question, gt, pred, max_retries=10, delay=2, idx=None):
    for attempt in range(max_retries):
        try:
            return model(question, gt, pred)
        except Exception as e:
            print(f"[Line {idx}] Attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
    print(f"[Line {idx}] All {max_retries} attempts failed.")
    return {"correct": False, "reason": "Failed after multiple retries"}

def safe_strip(x):
    return x.strip() if isinstance(x, str) else ""

def process_item(item, model, idx):
    question = safe_strip(item.get("question", ""))
    text_gt = safe_strip(item.get("answer_text", ""))
    text_pred = safe_strip(item.get("textqa_answer", ""))
    item["textqa_judge"] = safe_judge_with_retry(model, question, text_gt, text_pred, max_retries=10, delay=2, idx=idx)

    option_gt = safe_strip(item.get("answer", ""))
    option_pred = safe_strip(item.get("mcq_answer", ""))
    item["mcq_judge"] = option_judge(option_gt, option_pred)

    return item

def analyze(file_path):
    stats = {"textqa_judge_true": 0, "mcq_judge_true": 0, "total": 0}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("textqa_judge", False):
                stats["textqa_judge_true"] += 1
            if item.get("mcq_judge", False):
                stats["mcq_judge_true"] += 1
            stats["total"] += 1
    print(stats)
    print(round(stats["textqa_judge_true"] / stats["total"], 3))
    print(round(stats["mcq_judge_true"] / stats["total"], 3))
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stats": stats}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GPT-4o judging on VideoEval-Pro results')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input results file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save judged results')
    parser.add_argument('--model_name', type=str, default="gpt-4o-2024-08-06",
                        help='Path to GPT model')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='Number of threads for parallel processing')
    
    args = parser.parse_args()

    model = GPT4oJudge(model_name=args.model_name)

    processed_lines = 0
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            processed_lines = sum(1 for _ in f)

    all_items = []
    with open(args.input_path, "r", encoding="utf-8") as infile:
        for idx, line in enumerate(infile):
            if idx < processed_lines:
                continue
            try:
                item = json.loads(line)
                all_items.append((idx, item))
            except json.JSONDecodeError as e:
                print(f"[Line {idx}] Invalid JSON: {e}")
                continue

    with open(args.output_path, "a", encoding="utf-8") as outfile, ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for idx, item in all_items:
            futures.append(executor.submit(process_item, item, model, idx))
        
        completed = 0

        for future in tqdm(as_completed(futures), total=len(futures), desc="Judging with GPT-4o", dynamic_ncols=True):
            result_item = future.result()
            outfile.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            outfile.flush()

            completed += 1
            if completed % 100 == 0:
                print(f"Processed {completed} items, sleeping for 60 seconds...")
                time.sleep(60)

    print(f"Judging completed. Results saved to {args.output_path}")
    analyze(args.output_path)

