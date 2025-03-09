import os
import uuid
from datetime import datetime
from pathlib import Path
from re import VERBOSE
from typing import Any, Type

import matplotlib
import numpy as np
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun, YahooFinanceNewsTool

matplotlib.use("Agg")  # Set the backend to Agg before importing pyplot

import litellm
import pandas as pd
import requests
import torch
import vertexai
import yfinance
from chromadb import Documents, EmbeddingFunction, Embeddings
from chronos import BaseChronosPipeline
from crewai_tools import RagTool, TXTSearchTool, WebsiteSearchTool
from dotenv import load_dotenv
from langchain_google_vertexai import VertexAIEmbeddings
from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field

load_dotenv()
os.environ["OTEL_SDK_DISABLED"] =  'true'


VERBOSE = int(os.getenv("VERBOSE", 1)) == 1
os.environ["LITELLM_LOG"] = "DEBUG" if VERBOSE else "INFO"

os.environ["CHRONOS_MODEL"] = os.getenv("CHRONOS_MODEL", "amazon/chronos-t5-base")
os.environ["CHRONOS_NUM_SAMPLES"] = os.getenv("CHRONOS_NUM_SAMPLES", 20)
os.environ["CHRONOS_TEMPERATURE"] = os.getenv("CHRONOS_TEMPERATURE", 1.0)
os.environ["CHRONOS_TOP_K"] = os.getenv("CHRONOS_TOP_K", 50)
os.environ["CHRONOS_TOP_P"] = os.getenv("CHRONOS_TOP_P", 1.0)

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["JINA_API_KEY"] = os.getenv("JINA_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["BRAVE_API_KEY"] = os.getenv("BRAVE_SEARCH_API_KEY")
os.environ["OLLAMA_HOST"] = os.getenv("OLLAMA_HOST")
os.environ["GOOGLE_CLOUD_REGION"] = os.getenv("GOOGLE_CLOUD_REGION")
os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv("GOOGLE_CLOUD_PROJECT")

os.environ["LLM_TEMPERATURE"] = os.getenv("LLM_TEMPERATURE", 1.0)
os.environ["LLM_TOP_P"] = os.getenv("LLM_TOP_P", 1.0)
os.environ["LLM_TIMEOUT"] = os.getenv("LLM_TIMEOUT", 120)

logger.debug(
    f"gcp project: {os.getenv('GOOGLE_CLOUD_PROJECT')}, location: {os.getenv('GOOGLE_CLOUD_REGION')}"
)
vertexai.init(
    project=os.getenv("GOOGLE_CLOUD_PROJECT"),
    location=os.getenv("GOOGLE_CLOUD_REGION"),
)


litellm.vertex_project = os.getenv("GOOGLE_CLOUD_PROJECT")
litellm.vertex_location = os.getenv("GOOGLE_CLOUD_REGION")
litellm.set_verbose = False  # VERBOSE


class VertexAI_Embedder(EmbeddingFunction):
    embed = VertexAIEmbeddings(
        model_name="text-multilingual-embedding-002",  # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
        location=os.getenv("GOOGLE_CLOUD_REGION"),
    )

    def __call__(self, input: Documents) -> Embeddings:
        return self.embed.invoke(input)


rag_tool = RagTool(
    config=dict(
        llm=dict(
            provider="vertexai",
            config=dict(
                model="gemini-2.0-flash",
                temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
                top_p=float(os.getenv("LLM_TOP_P", 1.0)),
                # location=os.getenv("GOOGLE_CLOUD_REGION"),
            ),
        ),
        embedder=dict(
            provider="vertexai",
            config=dict(
                model="text-multilingual-embedding-002",
                # location=os.getenv("GOOGLE_CLOUD_REGION"),
            ),
        ),
    )
)

txt_rag_tool = TXTSearchTool(
    config=dict(
        llm=dict(
            provider="vertexai",
            config=dict(
                model="gemini-2.0-flash",
                temperature=float(os.getenv("LLM_TEMPERATURE", 1.0)),
                top_p=float(os.getenv("LLM_TOP_P", 1.0)),
                # location=os.getenv("GOOGLE_CLOUD_REGION"),
            ),
        ),
        embedder=dict(
            provider="vertexai",
            config=dict(
                model="text-multilingual-embedding-002",
                # location=os.getenv("GOOGLE_CLOUD_REGION"),
            ),
        ),
    )
)

web_rag_tool = WebsiteSearchTool(
    config={
        "embedding_model": {
            "provider": "vertexai",
            "config": {
                "model": "text-multilingual-embedding-002",
                # "location": os.getenv("GOOGLE_CLOUD_REGION"),
            },
        }
    },
)


class JinaReaderToolSchema(BaseModel):
    """Input for the JinaReaderTool"""

    website_url: str = Field(..., description="Mandatory website url to read")


class JinaReaderTool(BaseTool):
    name: str = "Read website content via Jina Reader"
    description: str = (
        "A tool that can be used to read the content behind a URL of a website via API of Jina Reader."
    )
    args_schema: Type[BaseModel] = JinaReaderToolSchema
    key: str = os.environ["JINA_API_KEY"]

    def _run(self, **kwargs: Any) -> Any:
        """
        api call:
curl https://r.jina.ai/https://example.com \
-H "Accept: text/event-stream" \
-H "Authorization: Bearer asdfasdfasdfasdfasdf" \
-H "X-Respond-With: readerlm-v2" \
-H "X-Return-Format: markdown" \
-H "X-Timeout: 60"
        """
        website_url = kwargs.get("website_url", None)
        if not website_url:
            logger.warning(f"website_url is required, not detected")
            return "website_url is required"
        try:
            logger.info(f"Retrieving content from {website_url} via 🐼 Jina Reader")
            txt = requests.get(
                f"https://r.jina.ai/{website_url}",
                headers={
                    "Authorization": f"Bearer {self.key}",
                    "X-Return-Format": "markdown",
                    "X-Timeout": "60",
                },
                timeout=60,
            ).text
            logger.success(f"Retrieved content from {website_url} via Jina Reader")
            return txt
        except Exception as e:
            if VERBOSE:
                logger.error(
                    f"Error while retrieving content from {website_url} via Jina Reader: {e}"
                )
            return f"Error while retrieving content from {website_url} via Jina Reader: {e}"


class YahooFinanceNewsTool(BaseTool):
    name: str = "Yahoo Finance News"
    description: str = "Useful for search-based queries on Yahoo Finance."
    search: YahooFinanceNewsTool = Field(default_factory=YahooFinanceNewsTool)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            logger.info(f"Retrieving news from 🦁Yahoo Finance for query {query}")
            res = self.search.run(query)
            logger.success(f"Retrieved news from Yahoo Finance for query {query}")
            return res
        except Exception as e:
            return f"Error performing search: {str(e)} on Yahoo Finance."


class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGoSearchTool"
    description: str = "Useful for search-based queries on DuckDuckGo."
    search: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun)

    def _run(self, query: str) -> str:
        """Execute the search query and return results"""
        try:
            logger.info(f"Retrieving news from DuckDuckGo 🦢 for query {query}")
            res = self.search.run(query)
            logger.success(f"Retrieved news from DuckDuckGo 🦢 for query {query}")
            return res
        except Exception as e:
            return f"Error performing search: {str(e)} on DuckDuckGo."


class GetCurrentTimeTool(BaseTool):
    name: str = "GetCurrentTimeTool"
    description: str = (
        """Useful for getting the current datetime. 
For questions or queries that are relevant to current time or date, 
this tool is useful. e.g. What is the current time? What is the current date?"""
    )

    def _run(self, **kwargs) -> str:
        logger.info("Retrieving current time 🧭")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return current_time


class StockSearchToolSchema(BaseModel):
    """Input for searching stock price based on the stock symbol, start date, and end date."""

    stock_symbols: list[str] = Field(
        ..., description="The stock symbol(s) to get the prices for."
    )
    start_date: str = Field(
        ..., description="The start date in the format 'YYYY-MM-DD'."
    )
    end_date: str = Field(..., description="The end date in the format 'YYYY-MM-DD'.")


def _create_random_filename(ext=".txt") -> str:
    return _create_random_name() + ext


def _create_random_name() -> str:
    return str(uuid.uuid4())


class StockSearchTool(BaseTool):
    name: str = "StockSearchTool"
    description: str = """Get and plot the stock prices for the given stock symbols between
    the start and end dates. The plot will have one line for each stock symbol. 

    Args:
        stock_symbols (str or list): The stock symbols to get the
        prices for.
        start_date (str): The start date in the format
        'YYYY-MM-DD'.
        end_date (str): The end date in the format 'YYYY-MM-DD'.

    Returns: tuple(pd.DataFrame, str, str):
        df (pandas.DataFrame): A pandas DataFrame with the stock prices for the given stock symbols indexed by date, with one column per stock symbol.  
        plot_fullpath (str): The filename (including path) of the plot of all stock prices queried.
        csv_path (str): The filename (including path) of the csv file of all stock prices queried.
    """
    args_schema: Type[BaseModel] = StockSearchToolSchema

    def _run(self, **kwargs) -> tuple[pd.DataFrame, str]:
        stock_symbols, start_date, end_date = (
            kwargs.get("stock_symbols"),
            kwargs.get("start_date"),
            kwargs.get("end_date"),
        )
        logger.info(
            f"Retrieving 📈 stock prices for {stock_symbols} between {start_date} and {end_date} 📉"
        )

        def _plot_stock_prices(stock_prices: pd.DataFrame, filename: str):
            """Plot the stock prices for the given stock symbols.
            Args:
                stock_prices (pandas.DataFrame): The stock prices for the given stock symbols.
                filename (str): The filename to save the plot to.

            """
            plt.figure(figsize=(10, 5))
            for column in stock_prices.columns:
                plt.plot(stock_prices.index, stock_prices[column], label=column)
            plt.title("Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend(stock_prices)
            plt.grid(True)
            plt.savefig(filename)

        stock_data = yfinance.download(stock_symbols, start=start_date, end=end_date)
        # pp(f"stock_data: {stock_data}")
        prices_df = stock_data.get("Close")
        # pp(f"Close: {prices_df}")

        plot_filename = _create_random_filename(ext=".regression.png")
        plot_fullpath = f"./output/{plot_filename}"
        Path(plot_fullpath).parent.mkdir(parents=True, exist_ok=True)

        csv_path = f"./output/{plot_filename}.csv"
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        prices_df.to_csv(csv_path)

        _plot_stock_prices(prices_df, plot_fullpath)
        logger.success(f"Complete retrieve stock: {stock_symbols}")
        return (
            prices_df,
            plot_fullpath,
            csv_path,
        )


class ForecastToolSchema(BaseModel):
    """Input for ForecastTool"""

    csv_path: str = Field(
        ...,
        description="A csv file that contains the dataframe, this the file path to the csv file, the dataframe is indexed by date, with one column of feature.",
    )

    prediction_length: int = Field(
        default=12,
        description="The prediction length.",
    )


class ForecastTool(BaseTool):
    name: str = "ForecastTool"
    description: str = (
        """Use time series forecasting model to forecast data based on historical data.

        Args:
            csv_path (str): A csv file that contains the dataframe, this the file path to the csv file, the dataframe is indexed by date, with one column of feature which is the historical data.
            prediction_length (int): The prediction length.
        Returns: 
            dfs (list[pd.DataFrame]): A list of dataframes of the forecasted data per series.
            plot_per_series_fullpath_list (list[tuple(str, str)]): A list of tuples of the filename (including path) of the plot images of the forecasted data per series and the csv file of the total forecasted data.
                                                                    Each csv represents each series and contains low-forecast, median-forecast, high-forecast the series.
    
        """
    )
    args_schema: Type[BaseModel] = ForecastToolSchema

    def _run(self, **kwargs) -> tuple:
        """Use time series forecasting model to forecast data based on historical data."""
        csv_path = kwargs.get("csv_path", None)
        if csv_path is None:
            raise ValueError("csv_path is required")

        # load csv including header
        df = pd.read_csv(csv_path, index_col=0)

        prediction_length = kwargs.get("prediction_length", 12)

        model = os.environ["CHRONOS_MODEL"]
        num_samples = int(os.environ["CHRONOS_NUM_SAMPLES"])
        temperature = float(os.environ["CHRONOS_TEMPERATURE"])
        top_k = int(os.environ["CHRONOS_TOP_K"])
        top_p = float(os.environ["CHRONOS_TOP_P"])

        pipeline = BaseChronosPipeline.from_pretrained(
            model,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )

        def _plot(forecast_index, column_name, df, low, median, high):
            plt.figure(figsize=(8, 4), facecolor="white")
            ax = plt.gca()
            ax.set_facecolor("white")

            plt.plot(df, color="royalblue", label="historical data")
            plt.plot(forecast_index, median, color="tomato", label="median forecast")
            plt.fill_between(
                forecast_index,
                low,
                high,
                color="tomato",
                alpha=0.3,
                label="80% prediction interval",
            )
            plt.legend()
            plt.grid(alpha=0.3, color="gray")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color("gray")
            ax.spines["left"].set_color("gray")

            plot_filename = _create_random_filename(ext=f".{column_name}.forecast.png")
            fullpath = f"./output/{plot_filename}"
            Path(fullpath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fullpath)

            return fullpath

        df_list = list()
        plot_per_series_fullpath_list = list()
        for idx, column in enumerate(df.columns):
            column_name = column
            logger.info(f"Processing column: {column_name} 📉 📈")
            context = torch.tensor(df.iloc[:, idx].values)

            forecast = pipeline.predict(
                context,
                prediction_length,
                num_samples=num_samples,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )  # forecast shape: [num_series, num_samples, prediction_length]

            low, median, high = np.quantile(
                forecast[0].numpy(),
                [
                    0.1,
                    0.5,
                    0.9,
                ],
                axis=0,
            )
            df_forecast = pd.DataFrame(
                columns=["low forecast", "median forecast", "high forecast"]
            )
            df_forecast["low forecast"] = low
            df_forecast["median forecast"] = median
            df_forecast["high forecast"] = high
            # set index name: 'Future business day'
            df_forecast.index.name = "Future business day"

            plot_filename = _create_random_filename(ext=f".{column_name}.forecast.png")
            context_csv_path = f"./output/{plot_filename}.csv"
            Path(context_csv_path).parent.mkdir(parents=True, exist_ok=True)
            df_forecast.to_csv(context_csv_path)

            df_list.append(df_forecast)
            plot_per_series_fullpath_list.append(
                (
                    _plot(
                        range(len(df), len(df) + prediction_length),
                        column_name,
                        df[column_name],
                        low,  # forecast part
                        median,  # forecast part
                        high,  # forecast part
                    ),
                    context_csv_path,
                )
            )
            logger.success(f"Complete processing column: {column_name} ✔")

        logger.success("Complete processing all columns 💨")
        return df_list, plot_per_series_fullpath_list


class SaveTextFileToolSchema(BaseModel):
    """Input for ForecastTool"""

    file_path: str = Field(
        ...,
        description="""The full path to the file to be saved, including filename.""",
    )

    content: str = Field(
        ...,
        description="The text content to be saved.",
    )


class SaveTextFileTool(BaseTool):
    name: str = "SaveTextFileTool"
    description: str = (
        """Save text file.

        Args:
            file_path (str): The full path to the file to be saved, including filename.
            content (str): The text content to be saved.
       
        """
    )
    args_schema: Type[BaseModel] = SaveTextFileToolSchema

    def _run(self, **kwargs) -> None:
        file_path = kwargs["file_path"]
        content = kwargs["content"]
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        # override file content if it exists
        if Path(file_path).exists():
            Path(file_path).unlink()
        with open(file_path, "w") as f:
            f.write(content)
