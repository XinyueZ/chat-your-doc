Chat Your Doc (Experimental)
=============

Chat Your Doc is an experimental project aimed at exploring various applications based on LLM. Although it is nominally a chatbot project, its purpose is broader. The project explores various applications using tools such as [LangChain](https://www.langchain.com/) or [scikit-learn LLM](https://github.com/iryna-kondr/scikit-llm). In the "Lab Apps" section, you can find many examples, including simple and complex ones. The project focuses on researching and exploring various LLM applications, while also incorporating other fields such as UX and computer vision. The "Lab App" section includes a table with links to various apps, descriptions, launch commands, and demos.

----

- [Setup](#setup)
    - [Some keys](#some-keys)
    - [Conda](#conda)
    - [Pip](#pip)   
- [Lab Apps](#lab-apps)
  - [Simple](#simple)
  - [Intermediate](#intermediate)
  - [Advanced](#advanced)
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

# Lab Apps

## Simple

| App | Models & Components|Description | Launch | Demo |
| --- | --- |--- | --- | --- |
| [i-ask.sh](simple/i-ask.sh)  |OpenAI http-request| Simply ask & answer via OpenAI API | `i-ask.sh "Who is Joe Biden?"` | ![](assets/screens/i-ask.gif) | 
| [chat_openai.py](simple/chat_openai.py)  |ChatOpenAI, LangChain | Just one chat session  | `streamlit run simple/chat_openai.py --server.port 8000 --server.enableCORS false` | ![](assets/screens/chat_openai.gif) | 
| [open_api_llm_app.py](simple/open_api_llm_app.py)  |OpenAI, LLMChain, LangChain| Use OpenAI LLM to answer simple question | `streamlit run simple/open_api_llm_app.py --server.port 8001 --server.enableCORS false` | ![](assets/screens/open_api_llm_app.gif) | 

## Intermediate

| App | Models & Components|Description | Launch | Demo |
| --- | --- |--- | --- | --- |
| [sim_app.py](intermediate/sim_app.py)  | Chroma, OpenAIEmbeddings, LangChain| Use the vector database to save file in chunks and retrieve similar content from the database | `streamlit run intermediate/sim_app.py --server.port 8002 --server.enableCORS false` | ![](assets/screens/sim_app.gif) | 
| [llm_chain_translator_app.py](intermediate/llm_chain_translator_app.py)  | ChatOpenAI, LLMChain, LangChain| Use LLMChain to do language translation | `streamlit run intermediate/llm_chain_translator_app.py --server.port 8003 --server.enableCORS false` | ![](assets/screens/llm_chain_translator_app.gif)  ![](assets/guide/llm_chain_translator_app.png) | 

## Advanced

| App |  Models & Components|Description | Launch | Demo |
| --- | --- |--- | --- | --- |
| [qa_chain_pdf_app.py](advanced/qa_chain_pdf_app.py)  |OpenAI, Chroma, load_qa_chain->BaseCombineDocumentsChain, LangChain| Ass info from PDF file, chat with it | `streamlit run advanced/qa_chain_pdf_app.py --server.port 8004 --server.enableCORS false` | ![](assets/screens/qa_chain_pdf_app.gif)  ![](assets/guide/qa_chain_pdf_app.png) | 


# Notes

> "These key notes can be very helpful in getting up to speed quickly. Look for them while you're learning and share them with others. These notes are especially useful when you're asking yourself questions like why, what, and how."

<details>
<summary><span style="font-weight: bold;">Prompt</span></summary>
<style>
  table {
    width: 100%;
    border-collapse: collapse;
  }
  td {
    padding: 8px;
    text-align: center;
  }
  img {
    width: 100%;
    height: auto;
  }
  @media screen and (max-width: 600px) {
    table, td, tr {
      display: block;
    }
    td {
      text-align: center;
      border-bottom: none;
    }
  }
</style>

<table>
  <tr>
    <td><img src="assets/notes/openai_moderation.png"></td>
    <td><img src="assets/notes/chain-of-thought-reasoning.png"></td>
    <td><img src="assets/notes/why_langchain_prompt_template.png"></td>
    <td><img src="assets/notes/memory.png"></td>
    <td><img src="assets/notes/memory_type_1.png"></td>
    <td><img src="assets/notes/memory_type_2.png"></td>
  </tr>
</table>
</details>