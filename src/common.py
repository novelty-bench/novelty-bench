from openai import AsyncOpenAI


def oai_client():
    with open("openai-api-key") as file:
        return AsyncOpenAI(api_key=file.read().strip())
