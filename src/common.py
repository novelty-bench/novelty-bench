import os

from openai import AsyncOpenAI


def oai_client():
    """Prefer the local key file, otherwise use the SDK's environment credentials."""
    try:
        with open("openai-api-key") as file:
            return AsyncOpenAI(api_key=file.read().strip())
    except FileNotFoundError:
        return AsyncOpenAI()


def google_project(project=None, default_project=None):
    project = project or os.getenv("GOOGLE_CLOUD_PROJECT") or default_project
    if not project:
        raise ValueError("Set --project or GOOGLE_CLOUD_PROJECT for Vertex inference")
    return project
