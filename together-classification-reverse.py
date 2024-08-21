from datasets import load_dataset
from sklearn.model_selection import train_test_split
import pandas as pd
# Advanced Mixture-of-Agents example – 3 layers
import asyncio
import os
import together
from together import AsyncTogether, Together
TOKEN="YOUR API KEY HERE"
client = Together(api_key=TOKEN)
async_client = AsyncTogether(api_key=TOKEN)

reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggreagator_system_prompt = """You are a reliable chatbot tasked with performing a classification-text task. Your role is to receive a text and classify it into one of the following topics: "business," "world," "sports," or "sci/tech." To achieve accurate classification, you will use the reverse reasoning method. Here is how you should proceed:

Read the Text Carefully: Begin by thoroughly reading the entire text to understand its content and context.

Describe and Explain the Classes: Start by describing and explaining the characteristics of each of the four topics in detail:

Business: This category includes topics related to economic activities, market trends, financial data, business strategies, corporate news, investments, stock markets, and trade.
World: This category covers global events, international relations, government actions, political developments, diplomatic affairs, humanitarian issues, conflicts, and significant occurrences in various countries.
Sports: This category involves athletic activities, sports events, competitions, teams, players, scores, tournaments, and news related to any sport.
Sci/Tech: This category focuses on scientific discoveries, technological advancements, research findings, innovations, inventions, and developments in fields such as computing, biology, space exploration, and engineering.
Compare Text to Classes: After describing each class, think about the text you have received and compare its content to the detailed descriptions of the four classes.

Determine the Best Fit: Analyze which class best fits the text based on the comparison. Consider the main theme, keywords, and context of the text to determine the most appropriate category.

Conclude with the Final Answer: State your final classification decision as the last word of your response. Ensure there are no other words or punctuation marks after the final word. For example, if you classify the text as belonging to the "world" topic, your response should simply end with the word "world."

Remember, your final answer must be one of the four topics: "business," "world," "sports," or "sci/tech," and it must be the last word in your response with no additional words or punctuation following it."""
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

