from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
# Advanced Mixture-of-Agents example – 3 layers
import asyncio
import os
import together
from together import AsyncTogether, Together
TOKEN="YOUR API TOKEN HERE"
client = Together(api_key=TOKEN)
async_client = AsyncTogether(api_key=TOKEN)

reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggreagator_system_prompt = """You are a reliable chatbot designed to perform text classification tasks. Your goal is to receive a text and classify it into one of the following categories: "business," "world," "sports," or "sci/tech." To ensure accurate classification, you will use a chain of thought process. Here are the steps you must follow:

Read and Understand the Text: Carefully read the provided text to understand its content, context, and main points.

Identify Key Elements: Identify key words, phrases, and themes that are relevant to each category. For example:

Business: Look for terms related to finance, markets, companies, economy, investments, etc.
World: Look for references to international events, politics, countries, global issues, etc.
Sports: Look for mentions of games, players, teams, scores, tournaments, etc.
Sci/Tech: Look for words related to technology, science, innovations, research, gadgets, etc.
Analyze the Context: Consider the context in which the key elements are used to ensure they align with the correct category. This helps in distinguishing between similar terms used in different contexts.

Compare and Contrast: Compare the identified elements with the characteristics of each category. This step helps to narrow down the possible categories.

Make a Decision: Based on the analysis, decide which category best fits the text. Ensure your reasoning is clear and logical.

Final Answer: Conclude your response with the category that you have classified the text into. The last word of your response must be the final answer and must be either "business," "world," "sports," or "sci/tech." There should be no additional words or punctuation after this final word.

Example Process:
Text: "The company reported a significant increase in its quarterly earnings, driven by strong sales in its international markets."
Key Elements: company, reported, significant increase, quarterly earnings, strong sales, international markets.
Context Analysis: The text discusses financial performance, sales, and markets.
Comparison: These elements are strongly related to business and finance.
Decision: The text best fits the "business" category.
Final Answer: business
Always follow this chain of thought process to ensure accurate and consistent classification. The final answer must be presented as the last word of your response, without any additional words or punctuation."""
layers = 3

def getFinalSystemPrompt(system_prompt, results):
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )

async def run_llm(model, user_prompt,prev_response=None):
    for sleep_time in [1, 2, 4]:
        try:
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            print("")
            break
        except together.error.RateLimitError as e:
            print("")
            await asyncio.sleep(sleep_time)
    return response.choices[0].message.content

async def main():
    
    # Define the labels
    labels = ["world","sports","business","sci/tech"]


    # Load the dataset and randomly select the texts
    ds = load_dataset("fancyzhx/ag_news")["test"]
    ds=[x for x in ds if type(x["label"])!=Ellipsis]
    df = pd.DataFrame(ds)
    texts,_=train_test_split(ds, train_size=100, stratify=df["label"], random_state=42)
    for text in texts:
        text["label"]=labels[text["label"]] 

    # Test the llm

    POINTS=0
    i=0
    for text in texts:
        """Run the main loop of the MOA process."""
        results = await asyncio.gather(*[run_llm(model,text["text"]) for model in reference_models])

        for _ in range(1, layers - 1):
            results = await asyncio.gather(
                *[run_llm(model,text["text"], prev_response=results) for model in reference_models]
            )

        finalStream = client.chat.completions.create(
            model=aggregator_model,
            messages=[
                {
                    "role": "system",
                    "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
                },
                {"role": "user", "content": text["text"]},
            ],
            stream=True,
        ) 
        response=[]
        for chunk in finalStream:
            response.append(chunk.choices[0].delta.content or "")
        response1=response[len(response)-2]
        response2=response[len(response)-3]
        if response1[len(response1)-3:]==text["label"][len(text["label"])-3:] or response2[len(response2)-3:]==text["label"][len(text["label"])-3:]:
            POINTS+=1
        i+=1
        print(i)
        print("SCORE SO FAR "+str(POINTS))
    print("SCORE: "+str(POINTS/100))
    

asyncio.run(main())



