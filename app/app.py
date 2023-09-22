from fastapi import FastAPI
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
import pandas as pd
from transformers import pipeline, BertForSequenceClassification, BertTokenizerFast
import json
from fastapi.applications import ASGIApp
from pydantic import BaseModel
import uvicorn
from io import StringIO
from fastapi.responses import StreamingResponse
import deduplicate
from deduplicate import Deduplication
import os
    
app = FastAPI()
#model_path = "text-classification-model"
#model = BertForSequenceClassification.from_pretrained(model_path)

import re
import pandas as pd

# Функция для удаления эмодзи из текста
def remove_emojis(text):
    # Паттерн для поиска эмодзи Unicode
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F"  # Эмодзи с лицевыми выражениями
                            u"\U0001F300-\U0001F5FF"  # Символы и пиктограммы
                            u"\U0001F680-\U0001F6FF"  # Транспорт и символы для путешествий
                            u"\U0001F1E0-\U0001F1FF"  # Флаги стран
                            u"\U00002702-\U000027B0"  # Декоративные значки
                            u"\U000024C2-\U0001F251" 
                            "]+", flags=re.UNICODE)
    # Заменить эмодзи на пустую строку
    return emoji_pattern.sub(r'', text)

def prepare_output_file(file_path):
    
    df = pd.read_csv(file_path)
    df['text'] = df['text'].apply(remove_emojis)
    deduplicator = Deduplication(df)
    
    news_df = deduplicator.remove_duplicates()  # Убираем дубликаты
    return news_df
    
    '''data = json.dumps(list_of_labels)
    data = json.loads(data)

    df = pd.DataFrame(data)
    df['label'] = df[0].apply(lambda x : x['label'])
    
    df['score'] = df[0].apply(lambda x : x['score'])
    df = df.drop([0],axis='columns') pd.concat([news_df,df], axis='columns',ignore_index=False)'''


@app.post("/process_csv/")
async def process_csv(files: list[UploadFile]):    
    # make some prediction
    #tokenizer = BertTokenizerFast.from_pretrained(model_path)
    #nlp = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)
    
    # [nlp(news[:512]) for news in df['text']],
    # crate an output csv file

    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")
    for file in files:
        # Сохраняем загруженный CSV-файл локально
        file_path = f"uploaded_files/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Обрабатываем CSV-файл и удаляем дубликаты
        deduplicated_df = prepare_output_file(file_path)
        
        # Сохраняем дедуплицированный DataFrame в новый файл CSV
        output_file_path = f"uploaded_files/deduplicated_{file.filename}"
        deduplicated_df.to_csv(output_file_path, index=False)
        
        return {"file_path": output_file_path}
    
@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}

if __name__ == '__main__':
    
    #config = uvicorn.Config(app,host="127.0.0.1", port=8000, log_level="info", workers=4)
    uvicorn.run(app,host="127.0.0.1", port=8000, log_level="info", workers=4)