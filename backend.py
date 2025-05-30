
#step 1: setup  pydantic model (schema validation )
from pydantic import BaseModel
from typing import List


class RecquestState(BaseModel):
    model_name:str
    model_provider:str
    system_prompt:str
    messages:List[str]
    allow_search:bool
# step 2 :setup AI agent from frontend Recquest 

from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent


ALLOWED_MODEL_NAMES = ["llama3-70b-8192", 
                       "mixtral-8x7b-32768",
                         "llama-3.3-70b-versatile",
                           "gpt-40-mini",
                             "llama-3.1-8b-instant", 
                             "gemma2-9b-it"
]

app=FastAPI(title="AI_Agent HelpDesk")

@app.post("/chat")
def chat_endpoint(request:RecquestState):
    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error":"Invalid model name.Kindly select valid AI model"}
    
    response=get_response_from_ai_agent(
       request.model_name,
        request.messages,
        request.allow_search,
      request.system_prompt,
       request.model_provider
    )
   
    return response 

if __name__=="__main__":
     import uvicorn
     uvicorn.run(app,host="127.0.0.1",port=8001)



 