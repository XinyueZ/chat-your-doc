import copy
import random
from typing import Tuple

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.messages import BaseMessage
from loguru import logger

random.seed(1024)


class PromptRepository:
    _chat: ChatOpenAI

    def __init__(self, key=None) -> None:
        self._chat = (
            ChatOpenAI(model="gpt-4-0613", verbose=True)
            if key is None
            else ChatOpenAI(model="gpt-4-0613", verbose=False, openai_api_key=key)
        )

    def prompt(self, prompt: list[BaseMessage]) -> BaseMessage:
        return self._chat(prompt)


class JokeProviderRepository:
    _sys_template: SystemMessagePromptTemplate
    _user_template: HumanMessagePromptTemplate
    _chat_template: ChatPromptTemplate
    _prompt_repository: PromptRepository

    _structured_output_parser: StructuredOutputParser

    def __init__(self, prompt_repository: PromptRepository) -> None:
        self._prompt_repository = prompt_repository

        sys_tmpl = """Create a mount of random jokes for me, answer me ONLY the jokes in response.
            The format instructions is {format_instructions}"""
        usr_tmpl = "Number of jokes to generate: {num_of_jokes}"
        self._sys_template = SystemMessagePromptTemplate.from_template(sys_tmpl)
        self._user_template = HumanMessagePromptTemplate.from_template(usr_tmpl)
        self._chat_template = ChatPromptTemplate.from_messages(
            [self._sys_template, self._user_template]
        )

        self._structured_output_parser = StructuredOutputParser.from_response_schemas(
            [
                ResponseSchema(
                    name="jokes",
                    description="The list of jokes you've provided",
                    type="List[string]",
                ),
            ]
        )

    def tell_joke(self, num_of_jokes) -> Tuple[BaseMessage, list[str]]:
        prompt = self._chat_template.format_messages(
            num_of_jokes=num_of_jokes,
            format_instructions=self._structured_output_parser.get_format_instructions(),
        )
        jokes = self._prompt_repository.prompt(prompt=prompt)

        structured_output = self._structured_output_parser.parse(jokes.content)
        list_output = structured_output["jokes"]
        # logger.debug("Background jokes: {}", len(list_output))

        return (jokes, list_output)


class JokeRatingRepository:
    _sys_template: SystemMessagePromptTemplate
    _user_template: HumanMessagePromptTemplate
    _chat_template: ChatPromptTemplate
    _prompt_repository: PromptRepository

    def __init__(self, prompt_repository: PromptRepository) -> None:
        sys_tmpl = "You should take a text (representing the joke to be rated) and return ONLY an integer from 1 to 10, which represents the rating of the joke."
        usr_tmpl = "Text: {text}"
        self._sys_template = SystemMessagePromptTemplate.from_template(sys_tmpl)
        self._user_template = HumanMessagePromptTemplate.from_template(usr_tmpl)
        self._chat_template = ChatPromptTemplate.from_messages(
            [self._sys_template, self._user_template]
        )
        self._prompt_repository = prompt_repository

    def rate_joke(self, joke: str) -> int:
        prompt = self._chat_template.format_messages(text=joke)
        rating = self._prompt_repository.prompt(prompt)
        return int(rating.content)


class Bot:
    _num_of_jokes: int
    _joke_repository: JokeProviderRepository
    _rating_repository: JokeRatingRepository

    def __init__(self, num_of_jokes: int) -> None:
        prompt_repository = PromptRepository()
        self._joke_repository = JokeProviderRepository(
            prompt_repository=prompt_repository
        )
        self._rating_repository = JokeRatingRepository(
            prompt_repository=prompt_repository
        )
        self._num_of_jokes = num_of_jokes

    def tell_joke(self) -> str:
        _, list_of_jokes = self._joke_repository.tell_joke(
            num_of_jokes=self._num_of_jokes
        )
        copy_of_jokes = copy.deepcopy(list_of_jokes)
        random.shuffle(copy_of_jokes)
        return random.choice(copy_of_jokes)

    def rate_joke(self, joke) -> int:
        return self._rating_repository.rate_joke(joke)


# The app can be started with two cases: 1. tell a joke, 2. rate a joke
# Commandline usage:
# 1. Tell one joke, random 8 (default) jokes at background and return one of them.
# python joke_bot.py --tell -num 8
# 2. Rate a joke, return the rating of the joke, an integer from 1 to 10.
# python joke_bot.py --rate "This is a joke"
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tell",
        action="store_true",
        help="Tell a joke, random 8 (default) jokes at background and return one of them.",
    )
    parser.add_argument(
        "--rate",
        type=str,
        help="Rate a joke, return the rating of the joke, an integer from 1 to 10.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=8,
        help="Number of jokes to generate, default is 8.",
    )
    args = parser.parse_args()

    bot = Bot(num_of_jokes=args.num)

    if args.tell and args.rate:
        logger.error("\n\nYou can't tell a joke and rate a joke at the same time.\n")
        parser.print_help()
    elif args.rate:
        logger.info(bot.rate_joke(args.rate))
    elif args.tell:
        logger.info(bot.tell_joke())
    else:
        parser.print_help()
