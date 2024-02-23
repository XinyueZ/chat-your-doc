from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama

llm = Ollama(
    base_url="http://localhost:11434",
    model="gemma:2b",
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()],
    ),
)
while True:
    query = input("Enter a query: ")
    llm.invoke(query)
    print()
