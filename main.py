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
          "content": "Опиши состав одежды построчно. Ответ напиши на русском языке."
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
