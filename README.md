Awesome LLMs applications and experiments
=============

Chat Your Doc is an experimental project aimed at exploring various applications based on LLM. Although it is nominally a chatbot project, its purpose is broader. The project explores various applications using tools such as [LangChain](https://www.langchain.com/), [LlamaIndex](https://www.llamaindex.ai/). In the "Lab Apps" section, you can find many examples, including simple and complex ones. The project focuses on researching and exploring various LLM applications, while also incorporating other fields such as UX and computer vision. The "Lab App" section includes a table with links to various apps, descriptions, launch commands, and demos.

----

- [Setup](#setup)
    - [Some keys](#some-keys)
    - [Conda](#conda)
    - [Pip](#pip)  
- [Storage](#storage) 
- [Popular solutions](#popular-solutions)
  - [How to chat with a document via vector database?](#how-to-chat-with-a-document-via-vector-database)
  - [Simple](#simple)
  - [Intermediate](#intermediate)
  - [Advanced](#advanced)
- [Notebooks](#notebooks)
- [Notes](#notes)

----
# Setup

## Some keys

For OpenAI, set API-KEY to your environment variables.

[Get OpenAI API-KEY](https://platform.openai.com/account/api-keys)

```bash
export OPENAI_API_KEY="sk-$TJHGFHJDSFDGAFDFRTHT§$%$%&§%&"
```

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

# Storage


ref: https://python.langchain.com/docs/modules/data_connection/vectorstores/

Chroma, FAISS

`pip install chromadb`

`pip install faiss-cpu`


# Popular solutions 

## Boilerplate codes of tokenization in LLM project, take away codes.

[LLM fine-tuning step: Tokenizing](https://medium.com/@teetracker/llm-fine-tuning-step-tokenizing-caebb280cfc2)

## How to chat with a document via vector database?

[Chat with your PDF （Streamlit Demo)](https://teetracker.medium.com/chat-with-your-pdf-streamlit-demo-eb2a3a2882a3)

## How do auto-annotation (object detection) on images?

[LLM, LangChain Agent, Computer Vision](https://medium.com/@teetracker/experimentation-llm-langchain-agent-computer-vision-0c405deb7c6e) 

## OpenAI Assistants, awesome, it manages a lot of things for you.

[How to Use OpenAI Assistants (API)](https://teetracker.medium.com/how-to-use-openai-assistants-api-2281d67486a0)

## Advanced RAG approaches, learn details how RAG works.
[Advanced RAG](https://medium.com/@teetracker/advanced-rag-228510e7ac77)

## LangChain App Template Example

[Experience with the LangChain App Template](https://teetracker.medium.com/experience-with-the-langchain-app-template-349fb47161c1)

> [wine price app, prompt and query wine price](simple/wine-price-app). The application has been created, just need to create data, install Poetry and run it. 

> Full instructions for creating a LangChain App Template can be found here: [LangChain App Template](https://python.langchain.com/docs/guides/deployments/template_repos)

# RAG, Vector and Summary pattern implementation

[Llama-Index: RAG with Vector and Summary](https://medium.com/@teetracker/llama-index-multi-vector-retriever-summary-9231137d3cab)
> Multi-Retrievers with Vector and Summary pattern implementation.

[Llama-Index: RAG with Vector and Summary by using Agent](https://medium.com/@teetracker/llama-index-rag-with-vector-and-summary-using-agent-551b4b7cef16)
> Agent for Vector and Summary pattern implementation.

[Create Agent from Scratch](https://teetracker.medium.com/building-an-agent-from-scratch-with-langchain-2e1d1ef2f57f) 

> Intuitively create an agent from scratch with LangChain.

[Create Agent from Scratch](https://medium.com/@teetracker/llama-index-building-an-agent-from-scratch-73be48f7f266) 

> Intuitively create an agent from scratch with  [LlamaIndex](https://www.llamaindex.ai/).

[Multi-Query Retrieval with RAG](https://teetracker.medium.com/langchain-llama-index-rag-with-multi-query-retrieval-4e7df1a62f83)

> Enhance query context with intermediate queries during RAG to improve information retrieval for the original query.

[Prio-reasoning](https://teetracker.medium.com/prio-reasoning-919fd6e90d86)

> Explain how prio-reasoning works in LangChain(qa-chain with refine) and LlamaIndex (MultiStepQueryEngine).

[RAG with Multi-Query pattern](https://medium.com/@teetracker/rag-with-multi-query-pattern-7272deb3401a) 

> Introduce the Multi-Query pattern in bundle and step-down solutions.

[LangGraph: hello,world!](https://teetracker.medium.com/langgraph-hello-world-d913677cc222)   

> Introduce the hello,world version use-case with LangGraph, just simple example to kick off the topic.

[Quick Tour: Ollama+Gemma+LangChain](https://teetracker.medium.com/quick-tour-ollama-gemma-langchain-ea28e5314256) 

> Use ollama to deploy and use Gemma with langchain on local machine.

## Simple

| App | Description | Launch | Demo |
| --- |--- | --- | --- |
| [i-ask.sh](simple/i-ask.sh)  | Simply ask & answer via OpenAI API | `i-ask.sh "Who is Joe Biden?"` | ![](assets/screens/i-ask.gif) | 
| [chat_openai.py](simple/chat_openai.py)  | Just one chat session  | `streamlit run simple/chat_openai.py --server.port 8000 --server.enableCORS false` | ![](assets/screens/chat_openai.gif) | 
| [open_api_llm_app.py](simple/open_api_llm_app.py)   | Use OpenAI LLM to answer simple question | `streamlit run simple/open_api_llm_app.py --server.port 8001 --server.enableCORS false` | ![](assets/screens/open_api_llm_app.gif) | 
| [read_html_app.py](simple/read_html_app.py)  | Get html content and chunk | `streamlit run simple/read_html_app.py --server.port 8002 --server.enableCORS false` | ![](assets/screens/read_html_app.gif) | 
| 💥 [chatbot.py](simple/chatbot.py)   | Basic chatbot | `streamlit run simple/chatbot.py --server.port 8003 --server.enableCORS false` | ![](assets/screens/chatbot.gif) ![](assets/guide/chatbot.png)| 
| [retriever.py](simple/retriever.py)  | Use concept of retriever and LangChain Expression Language (LCEL) | `streamlit run simple/retriever.py --server.port 8004 --server.enableCORS false` | ![](assets/notes/retriever.png) ![](assets/screens/retriever.gif)| 
| [hello_llamaindex.py](simple/hello_llamaindex.py) | A very simple [LlamaIndex](https://www.llamaindex.ai/) to break ice of the story. | `streamlit run simple/hello_llamaindex.py --server.port 8005 --server.enableCORS false` | | 
| [llamaindex_context.py](simple/llamaindex_context.py)  | A simple app of [LlamaIndex](https://www.llamaindex.ai/), introduce of context for configuration, StorageContext, ServiceContext. | `streamlit run simple/llamaindex_context.py --server.port 8006 --server.enableCORS false` | | 
| [llamaindex_hub_simple.py](simple/llamaindex_hub_simple.py)   | A simple app of [LlamaIndex](https://www.llamaindex.ai/), introduce of load stuff from [https://llamahub.ai/](LlamaHub). | `streamlit run simple/llamaindex_hub_simple.py --server.port 8007 --server.enableCORS false` | | 
| 💥 [prio_reasoning_context.py](simple/prio_reasoning_context.py)   | A simple app that based on RAG with Prio-Reasoning pattern in  [LlamaIndex](https://www.llamaindex.ai/) or LangChain.| `streamlit run simple/prio_reasoning_context.py --server.port 8008 --server.enableCORS false` | [read](https://teetracker.medium.com/https://teetracker.medium.com/prio-reasoning-919fd6e90d86) ![](assets/screens/prio_reasoning_context.gif) ![](assets/guide/prio_reasoning_context_langchain.webp) ![](assets/guide/prio_reasoning_context_llamaindex.webp) | 
| 💥 [ollama_gemma.py](simple/ollama_gemma.py)   | A [Ollama](https://ollama.com/library/gemma) for Gemma integration with the langchain.|  | [read](https://teetracker.medium.com/quick-tour-ollama-gemma-langchain-ea28e5314256) ![](assets/screens/ollama_gemma.gif)  | 

## Intermediate

| App | Description | Launch | Demo |
| --- | --- | --- | --- |
| 💥 [sim_app.py](intermediate/sim_app.py)  | Use the vector database to save file in chunks and retrieve similar content from the database | `streamlit run intermediate/sim_app.py --server.port 8002 --server.enableCORS false` | ![](assets/screens/sim_app.gif) | 
| [llm_chain_translator_app.py](intermediate/llm_chain_translator_app.py) | Use LLMChain to do language translation | `streamlit run intermediate/llm_chain_translator_app.py --server.port 8003 --server.enableCORS false` | ![](assets/screens/llm_chain_translator_app.gif)  ![](assets/guide/llm_chain_translator_app.png) | 
| [html_summary_chat_app.py](intermediate/html_summary_chat_app.py)   | Summary html content | `streamlit run intermediate/html_summary_chat_app.py --server.port 8004 --server.enableCORS false` | ![](assets/screens/html_summary_chat_app.gif) | 
| 💥 [html_2_json_app.py](intermediate/html_2_json_app.py) | Summary html keypoints into keypoint json | `streamlit run intermediate/html_2_json_app.py --server.port 8005 --server.enableCORS false` | ![](assets/screens/html_2_json_app.png) | 
| [assistants.py](intermediate/assistants.py)   | Use [OpenAI Assistants API](https://platform.openai.com/docs/assistants) in different ways | `streamlit run intermediate/assistants.py --server.port 8006 --server.enableCORS false` | [read](https://teetracker.medium.com/how-to-use-openai-assistants-api-2281d67486a0) ![](assets/guide/OpenAI_Assistant_Chat_And_Completions.jpeg) ![](assets/guide/OpenAI_Assistant_Function_And_Tool.jpeg) ![](assets/screens/assistants1.gif) ![](assets/screens/assistants2.gif) ![](assets/screens/assistants3.gif) | 

## 💥 Advanced

| App | Description | Launch | Demo |
| --- | --- | --- | --- |
|  [qa_chain_pdf_app.py](advanced/qa_chain_pdf_app.py)  | Ask info from PDF file, chat with it | `streamlit run advanced/qa_chain_pdf_app.py --server.port 8004 --server.enableCORS false` | ![](assets/screens/qa_chain_pdf_app.gif)  ![](assets/guide/qa_chain_pdf_app.png) | 
|  [faiss_app.py](advanced/faiss_app.py)  | Ask info from a internet file, find similar docs and answer with  **VectorDBQAWithSourcesChain** | `streamlit run advanced/faiss_app.py --server.port 8005 --server.enableCORS false` | ![](assets/screens/faiss_app.gif)  ![](assets/guide/faiss_app.png) | 
|  [html_2_json_output_app.py](advanced/html_2_json_output_app.py)  | Load html content and summary into json objects | `streamlit run advanced/html_2_json_output_app.py --server.port 8006 --server.enableCORS false` | ![](assets/screens/html_2_json_output_app.png)  ![](assets/guide/html_2_json_output_app.png) | 
|  [joke_bot.py](advanced/joke_bot.py)  | Prompt engineering to get one random joke or rate one joke | `python advanced/joke_bot.py --rate "Why couldn't the bicycle stand up by itself? It was two tired."` or `python advanced/joke_bot.py --tell --num 4` | ![](assets/screens/joke_bot.gif) ![](assets/guide/joke_bot.png) | 
|  [chat_ur_docs.py](advanced/chat_ur_docs.py)   | Chat with documents freely | `streamlit run advanced/chat_ur_docs.py --server.port 8004 --server.enableCORS false` |[read](https://medium.com/@teetracker/chat-with-your-pdf-streamlit-demo-eb2a3a2882a3)  ![](assets/notes/chat-doc-flow.jpeg) | 
|  💥 [image_auto_annotation.py](advanced/image_auto_annotation.py)   | Use LLM, LangChain Agent and GroundingDINO to detect objects on images freely (auto-annotation) | `streamlit run advanced/image_auto_annotation.py --server.port 8006 --server.enableCORS false` | [read](https://medium.com/@teetracker/experimentation-llm-langchain-agent-computer-vision-0c405deb7c6e) ![](assets/screens/image_auto_annotation.gif)  | 
|  [adv_rag.py](advanced/adv_rag.py)  | Advanced RAG approaches, use partition_pdf to extract texts and tables and analyze them | `streamlit run advanced/adv_rag.py --server.port 8007 --server.enableCORS false` | [read](https://medium.com/@teetracker/advanced-rag-228510e7ac77) | 
|  [llamaindex_vector_summary_retriever.py](advanced/llamaindex_vector_summary_retriever.py)    | Use [LlamaIndex](https://www.llamaindex.ai/) to apply vectory/summary pattern by using multi retrievers | `streamlit run advanced/llamaindex_multi_vector_summary.py --server.port 8008 --server.enableCORS false` |  [read](https://medium.com/@teetracker/llama-index-multi-vector-retriever-summary-9231137d3cab)   ![](assets/guide/vector_summary_retriever.jpg) | 
|  [llamaindex_vector_summary_agent.py](advanced/llamaindex_vector_summary_agent.py)  | Use [LlamaIndex](https://www.llamaindex.ai/) to apply vectory/summary pattern by using agent | `streamlit run advanced/llamaindex_multi_vector_summary_agent.py --server.port 8009 --server.enableCORS false` | [read](https://medium.com/@teetracker/llama-index-rag-with-vector-and-summary-using-agent-551b4b7cef16)   ![](assets/guide/vector_summary_agent.png)  | 
|  💥 [multi_queries.py](advanced/multi_queries.py)  | Use [LlamaIndex](https://www.llamaindex.ai/) and LangChain to apply the multi-queries pattern, including build method of the LangChain, and custom retriever based solution in Llama-Index, also the other sub-query based solutions | `streamlit run advanced/multi_queries.py --server.port 8010 --server.enableCORS false` | [read](https://medium.com/@teetracker/rag-with-multi-query-pattern-7272deb3401a)    ![](assets/guide/multi_queries_in_bundle.webp)   ![](assets/guide/multi_queries_in_step_down.webp)  | 

# Notebooks

| Notebook |Description |  Demo |
| --- | --- | --- |
| [audio2text2LLM.ipynb](notebooks/audio2text2LLM.ipynb)  | Basic audio to text and summary | ![](assets/guide/audio2text2LLM.png)| 
| [audio2text2music.ipynb](notebooks/audio2text2music.ipynb)   | [audiocraft](https://github.com/facebookresearch/audiocraft), Whisper,  automatic-speech-recognition, speech to text, generate music by the text, synthesis speech+BGM |  ![](assets/guide/audio2text2music.png)| 
| [image_description.ipynb](notebooks/image_description.ipynb)   | [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base), [blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large), a use-case to get the image description | | 
| [image_desc2music.ipynb](notebooks/image_desc2music.ipynb)   | [audiocraft](https://github.com/facebookresearch/audiocraft) [blip-image-captioning-base](https://huggingface.co/Salesforce/blip-image-captioning-base), [blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large), a use-case to get the image description and generate music based on the image | | 
| [langchain_agent_scratch.ipynb](notebooks/langchain_agent_scratch.ipynb)  | Create the agent from scratch in langchain  | [read](https://teetracker.medium.com/building-an-agent-from-scratch-with-langchain-2e1d1ef2f57f)   ![](assets/guide/langchain_agent_scratch.webp) | 
| [llamaindex_agent_from_scratch.ipynb](notebooks/llamaindex_agent_from_scratch.ipynb)  | Create the agent from scratch with [LlamaIndex](https://www.llamaindex.ai/)  | [read](https://teetracker.medium.com/llama-index-building-an-agent-from-scratch-73be48f7f266)   ![](assets/guide/llamaindex_agent_scratch.jpg) | 
| [llamaindex_vector_summary_retriever.ipynb](notebooks/llamaindex_vector_summary_retriever.ipynb)  | Use [LlamaIndex](https://www.llamaindex.ai/) to apply vectory/summary pattern by using multi retrievers | [read](https://medium.com/@teetracker/llama-index-multi-vector-retriever-summary-9231137d3cab)   ![](assets/guide/vector_summary_retriever.jpg) | 
| [llamaindex_vector_summary_agent.ipynb](notebooks/llamaindex_vector_summary_agent.ipynb)  | Use [LlamaIndex](https://www.llamaindex.ai/) to apply vectory/summary pattern by using agent | [read](https://medium.com/@teetracker/llama-index-rag-with-vector-and-summary-using-agent-551b4b7cef16)    ![](assets/guide/vector_summary_agent.png) | 
| [multi_queries_retrieval.ipynb](notebooks/multi_queries_retrieval.ipynb)  | Use [LlamaIndex](https://www.llamaindex.ai/) and LangChain to apply mutli-query pattern for RAG | [read](https://teetracker.medium.com/langchain-llama-index-rag-with-multi-query-retrieval-4e7df1a62f83)    ![](assets/guide/multi_queries.png) | 
| [yolo8_world_and_openai_vision.py](notebooks/yolo8_world_and_openai_vision.py)  | Use [YoLoV8 World](https://docs.ultralytics.com/zh/models/yolo-world/#predict-usage) and OpenAI Vision together to enchance image auto-annotation |   ![](assets/screens/yolo8world_openai_vision.png) | 
| [langgraph_helloworld.py](notebooks/langgraph_helloworld.ipynb)  | hello,world version of langgraph |   [read](https://teetracker.medium.com/langgraph-hello-world-d913677cc222)    ![](assets/guide/langgraph_helloworld_1.webp) ![](assets/guide/langgraph_helloworld_2.webp) | 
