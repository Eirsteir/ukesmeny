FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8080
 
EXPOSE 8080
 
RUN apt-get update && apt-get install -y git
  
RUN pip install langchain chainlit lancedb pandas openai tiktoken
 
WORKDIR app/
 
COPY ./data/preprocessed/recipes.pkl /app/data/preprocessed/recipes.pkl
COPY ./app.py /app/app.py
COPY ./chainlit.md /app/chainlit.md
 
ENTRYPOINT ["chainlit", "run", "app.py"]
