import fastapi
import uvicorn
import openai
import os
from pydantic import BaseModel


client = openai.OpenAI()
app = fastapi.FastAPI()


class TextRequest(BaseModel):
    text: str


@app.post("/summarize", name = "Проанализировать бирку", tags = ["GPT4"])
def generate(request: TextRequest):
    response = client.chat.completions.create(
      model="gpt-4-1106-preview",
      messages=[
        {
          "role": "system",
          "content": """Тебе будет предоставлен не структурированный текст с бирок от одежды на разных языках. 
                        Твоя задача определить все типы тканей которые упоминаются в тексте. 
                        Результат преобразовать в формат CSV где первый столбец будет названием части одежды, второй столбец название ткани и третий столбец процент содержания ткани. 
                        Ответ выдать на русском языке. Не добавлять свои комментарии к выводу. Не писать ничего лишнего."""
        },
        {
          "role": "user",
          "content": request.text
        }
      ],
      temperature=0,
      max_tokens=1024,
      top_p=0,
      frequency_penalty=0,
      presence_penalty=0
    )
    return {"result": response.choices[0].message.content}


if __name__ == "__main__":
    config = uvicorn.Config("main:app",
                            port=8040,
                            host="0.0.0.0",
                            log_level="info",
                            workers=os.cpu_count())

    server = uvicorn.Server(config)
    server.run()
