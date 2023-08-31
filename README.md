Chat Your Doc (Experimental)
=============

Chat Your Doc is an experimental project aimed at exploring various applications based on LLM. Although it is nominally a chatbot project, its purpose is broader. The project explores various applications using tools such as [LangChain](https://www.langchain.com/) or [scikit-learn LLM](https://github.com/iryna-kondr/scikit-llm). In the "Lab Apps" section, you can find many examples, including simple and complex ones. The project focuses on researching and exploring various LLM applications, while also incorporating other fields such as UX and computer vision. The "Lab App" section includes a table with links to various apps, descriptions, launch commands, and demos.

----
# Setup

## Conda

```bash
conda env create -n chat-ur-doc -f environment.yml
conda activate chat-ur-doc
```

## Pip

```bash
pip install -r requirements.txt
``` 
----

# Lab Apps

| App | Level| Models & Components|Description | Launch | Demo |
| --- | --- |--- |--- | --- | --- |
| [open_api_llm_app.py](simple/open_api_llm_app.py)  |simple|OpenAI, LLMChain, LangChain| Use OpenAI LLM to answer simple question | `streamlit run simple/open_api_llm_app.py --server.port 8001 --server.enableCORS false` | ![](assets/screens/open_api_llm_app.gif) | 
| [sim_app.py](intermediate/sim_app.py)  |intermediate|Chroma, OpenAIEmbeddings, LangChain| Use the vector database to save file in chunks and retrieve similar content from the database | `streamlit run intermediate/sim_app.py --server.port 8002 --server.enableCORS false` | ![](assets/screens/sim_app.gif) | 
| [llmchain_translator_app.py](intermediate/llmchain_translator_app.py)  |intermediate|ChatOpenAI, LLMChain, LangChain| Use LLMChain to do language translation | `streamlit run intermediate/llmchain_translator_app.py --server.port 8003 --server.enableCORS false` | ![](assets/screens/llmchain_translator_app.gif)  ![](assets/guide/llmchain_translator_app.png) | 
