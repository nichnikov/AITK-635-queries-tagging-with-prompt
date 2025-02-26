import os
import re
import json
import pandas as pd
from datetime import datetime
from itertools import groupby
from operator import itemgetter
import logging
from dotenv import load_dotenv
from llm_agent import AIAgent, OpenAIClient
 
load_dotenv()

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIAgent")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API ключ не найден. Убедитесь, что файл .env содержит переменную OPENAI_API_KEY.")


# Инициализация клиента и агента
openai_client = OpenAIClient(api_key=api_key)
ai_agent = AIAgent(ai_client=openai_client, logger=logger)

with open(os.path.join("data", "promts.json")) as jf:
    prmt = json.load(jf)
print(prmt)

fns = ["bss.csv", "kss.csv", "uss.csv"]
for fn in fns:
    df = pd.read_csv(os.path.join("data", fn), sep="\t")
    
    for col in df.columns:
        df.rename(columns={col: re.sub(r"\s+", "", col)}, inplace=True)
    
    if "createdon" in df.columns:
        df["createdon"] = df["createdon"].apply(lambda x: re.sub(r"\/", ".", x))
    if "request_string_normal" in df.columns:
        df["request_string"] = df["request_string"].apply(lambda x: re.sub(r"\s+", " ", x))
    if "request_string_normal" in df.columns:
        df["request_string_normal"] = df["request_string_normal"].apply(lambda x: re.sub(r"\s+", " ", x))
    if "pub" in df.columns:
        df["pub"] = df["pub"].apply(lambda x: re.sub(r"\s+", " ", x))

    data_dics = df.to_dict(orient="records")

    num = 1
    results = []
    for d in data_dics:
        query = d["request_string"]
        start_time = datetime.now() 
        prompt = prmt["promt"].format(str(query))

        if num > 50000:
            break
        num += 1 

        response = ai_agent(prompt, 
                            model="openai/gpt-4o-mini", 
                            temperature=0.7, 
                            max_tokens=3000)

        res_list = response.split("\n")

        for l in res_list:
            if len(re.findall(r"\|", l)) == 2:
                splt_r = l.split("|")
                results.append({
                                "init_query": query, 
                                "rep_query": splt_r[0], 
                                "sys_num": splt_r[1],
                                "sys_name": splt_r[2]
                                })

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join("results", fn), sep="\t", index=False)
        end_time = datetime.now() 
        logger.info(f"Файл: {fn} обработано {num} запросов из {len(data_dics)} за {end_time - start_time} секунд")
