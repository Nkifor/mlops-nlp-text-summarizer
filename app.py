from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from pydantic import BaseModel

from mlops_NLP_Text_Summarization.pipeline.prediction import ModelEvaluationTrainingPipeline


text:str = "What is Text Summarization?"

class Summary_args(BaseModel):
    length_penalty:float
    number_of_beams:int
    max_length:int


default_arguments = Summary_args(length_penalty= 0.8, number_of_beams=6, max_length= 500)

description = """

#### Text Summarization API allows you to summarize your text using fine tuned transformer model. 📑

## Parameters

To maximize the flexibility of the API, the following parameters are available:
* **text**: The text to summarize.
* **length_penalty**: Exponential penalty to the length. 1.0 means no penalty.
Set to values < 1.0 in order to encourage the model to generate shorter summaries.
* **number_of_beams**: Number of beams for beam search. Must be between 1 and infinity.
The number_of_beams parameter determines how many of these alternative sequences are kept at each step.
A higher num_beams value means that more alternative sequences are explored, which can lead to better results but also requires more computation
* **max_length**: The maximum length of the sequence to be generated. Between 0 and infinity.

#### Terms of Use
Feel free to test the API as much as you want.
"""


app = FastAPI(
    title="Summarization API",
    description=description,
    version="0.0.1",
    contact={
        "name": "Nkifor",
        "url": "https://github.com/Nkifor",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    }
)

@app.get("/", tags=["documentation"])
async def index():
    return RedirectResponse(url="/docs")



#@app.get("/train")
#async def training():
#    try:
#        os.system("python main.py")
#        return Response("Training successful !!")
#
#    except Exception as e:
#        return Response(f"Error Occurred! {e}")




@app.post("/predict", tags=["prediction"])
async def summarization(text, args:Summary_args = default_arguments):
    try:

        obj = ModelEvaluationTrainingPipeline()
        text = obj.predict(text, args.length_penalty, args.number_of_beams, args.max_length)
        return text
    except Exception as e:
        raise e


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)