# Databricks notebook source
# MAGIC %md
# MAGIC #### Using Autogen Assistant Agent to search for and retrieve content from Azure AI Search.

# COMMAND ----------

import json
import os
import requests
import autogen
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import (
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

# COMMAND ----------

# Import Cognitive Search index ENV
AZURE_SEARCH_SERVICE = "aisearch02"
AZURE_SEARCH_INDEX = "aisearch-index-recursive"
AZURE_SEARCH_KEY = dbutils.secrets.get(scope="myscope", key="aisearch-adminkey")
AZURE_SEARCH_API_VERSION = "2024-06-01-preview"
AZURE_SEARCH_SEMANTIC_SEARCH_CONFIG = "my-semantic-config"
AZURE_SEARCH_SERVICE_ENDPOINT = dbutils.secrets.get(scope="myscope", key="aisearch-endpoint")

# COMMAND ----------

from azure.search.documents import SearchClient

search_credential = AzureKeyCredential(dbutils.secrets.get(scope="myscope", key="aisearch-adminkey"))
search_endpoint = dbutils.secrets.get(scope="myscope", key="aisearch-endpoint")
index_name = "aisearch-index-recursive"

search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=search_credential)

# COMMAND ----------

llm_config = {
    "config_list": [
        {
            "model": dbutils.secrets.get(scope="myscope", key="aoai-deploymentname"),
            "api_key": dbutils.secrets.get(scope="myscope", key="aoai-api-key"),
            "base_url": dbutils.secrets.get(scope="myscope", key="aoai-endpoint"),
            "api_type": "azure",
            "api_version": "2024-02-15-preview",
        },
    ]
}

gpt4_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": llm_config["config_list"],
    "timeout": 120
}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the tool/function search_retrieval that will interact with the Azure AI Search service.

# COMMAND ----------

def search_retrieval(user_input:str) -> str:
        """
        Search and retrieve answers from Azure AI Search.
        Returns:
            str
        """
        query = user_input
        search_client = SearchClient(endpoint=search_endpoint, index_name=index_name, credential=search_credential)
        vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=5, fields="vector", exhaustive=True)

        r = search_client.search(  
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='my-semantic-config',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=1
    )
        #query_result = results.get_answers()[0].text
        results = [doc["content"].replace("\n", "").replace("\r", "") for doc in r]
        content = "\n".join(results)
        return content

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define the AssistantAgent and UserProxyAgent instances, and register the search function to them.

# COMMAND ----------

ai_search_agent = AssistantAgent(
    name="AISearch",
    system_message="You are a helpful AI agent."
    "You can help with Azure AI Search service."
    "Return TERMINATE when the task is done",
    llm_config=gpt4_config
)

user_proxy = UserProxyAgent(
    name="User",
    llm_config=False,
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="NEVER",
    code_execution_config=False
)

register_function(
    f=search_retrieval,
    caller=ai_search_agent,
    executor=user_proxy,
    name="search_retrieval",
    description="A tool or function for search retrieval from Azure AI Search"
)

# COMMAND ----------

await user_proxy.a_initiate_chat(recipient=ai_search_agent, message="Search for 'What determines the venue of a legal action brought against Northwind Health?' in the above defined index")
