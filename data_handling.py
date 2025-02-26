import os
import re
import json
import pandas as pd
from itertools import groupby
from operator import itemgetter
from llm_agent import ai_agent

with open(os.path.join("data", "promts.json")) as jf:
    prmt = json.load(jf)

print(prmt["promt"])

fns = ["отписка.csv"]


for fn in fns:
    df = pd.read_csv(os.path.join("data", fn), sep="\t")
    
    for col in df.columns:
        df.rename(columns={col: re.sub(r"\s+", "", col)}, inplace=True)
    
    df["createdon"] = df["createdon"].apply(lambda x: re.sub(r"\/", ".", x))
    df["request_string"] = df["request_string"].apply(lambda x: re.sub(r"\s+", " ", x))
    df["request_string_normal"] = df["request_string_normal"].apply(lambda x: re.sub(r"\s+", " ", x))
    df["pub"] = df["pub"].apply(lambda x: re.sub(r"\s+", " ", x))

    print(df.info())
    data_dics = df[:5000].to_dict(orient="records")
    print(data_dics[:10])
    data_dics.sort(key=itemgetter('bitrixid'))
    
    dict_of_queries = {int(k): [d["createdon"] + "|" + d["request_string"] for d in list(g)] for k, g in groupby(data_dics, itemgetter("bitrixid"))}
    

    '''
    response = ai_agent(prompt, 
                       model="openai/gpt-4o-mini", 
                       temperature=0.7, 
                       max_tokens=3000)'''

    # print(dict_of_queries)
    k = 1
    for i in dict_of_queries:
        promt = """
                Вопросы пользователя:
                {}
                {}
                """.format(prmt["promt"], "\n".join(dict_of_queries[i]))
        print(promt)
        k +=1 
        if k > 5:
            break


    # print(dict_of_queries[:5])