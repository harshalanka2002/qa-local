

\## qa-local README.md

Same but ports and local mode:



```md

\# qa-local (CS2)



Frontend: Gradio  

Backend: FastAPI  

Mode: Local (Transformers pipeline)



\## Ports (local / VM)

\- Backend: 9044

\- Frontend: 7044



\## Run backend

```bash

pip install -r requirements.txt

uvicorn backend.main:app --host 0.0.0.0 --port 9044

