import lightning as L
import lightning.app.frontend as frontend
import streamlit as st

def main():
    st.title("llamaindex_adaptive_rag")

def your_streamlit_app(lightning_app_state):
    st.write('hello world')

class LitStreamlit(L.app.LightningFlow):
    def configure_layout(self):
        return frontend.StreamlitFrontend(render_fn=main)

class LitApp(L.app.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_streamlit = LitStreamlit()

    def run(self):
        self.lit_streamlit.run()

    def configure_layout(self):
        tab1 = {"name": "home", "content": self.lit_streamlit}
        return tab1

app = L.app.LightningApp(LitApp())
