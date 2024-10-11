
from llama_index.core.settings import Settings
#from llama_index.llms.cohere import Cohere
from llama_index.llms.openai import OpenAI
#from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.query_pipeline import QueryPipeline


def setup_llm(provider, model, api_key, **kwargs):
    """
    Configures the LLM (Language Learning Model) settings.

    Parameters:
    - api_key (str): The API key for authenticating with the LLM service.
    - model (str): The model identifier for the LLM service.
    """
        
    
    if provider == "openai":
        Settings.llm = OpenAI(model=model, api_key=api_key, **kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}. Pick 'openai'.")
    
    
def setup_embed_model(provider, **kwargs):
    """
    Configures the embedding model settings.

    Parameters:
        provider (str): The LLM provider ("openai").

    Raises:
        ValueError: If an invalid provider is specified.
    """
    
    if provider == "openai":
        Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large")
    else:
        raise ValueError(f"Invalid provider: {provider}. Pick 'openai'.")

def setup_vector_store(qdrant_url, qdrant_api_key, collection_name, enable_hybrid=False):
    """
    Creates and returns a QdrantVectorStore instance configured with the specified parameters.

    Parameters:
    - qdrant_url (str): The URL for the Qdrant service.
    - qdrant_api_key (str): The API key for authenticating with the Qdrant service.
    - collection_name (str): The name of the collection to be used in the vector store.

    Returns:
    - QdrantVectorStore: An instance of QdrantVectorStore configured with the specified Qdrant client
    """
    client = QdrantClient(location=qdrant_url, api_key=qdrant_api_key)
    aclient = AsyncQdrantClient(location=qdrant_url, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(client=client, aclient=aclient, collection_name=collection_name, enable_hybrid=enable_hybrid)
    return vector_store

def create_index(from_where, embed_model=Settings.embed_model, **kwargs):
    """
    Creates and returns a VectorStoreIndex instance configured with the specified parameters.

    Parameters:
    **kwargs: Additional keyword arguments for configuring the index, such as:
        - embed_model: The embedding model to be used in the index.
        - vector_store: The vector store to be used in the index.
        - nodes: The nodes to be used in the index.
        - storage_context: The storage context to be used in the index.

    Returns:
    - VectorStoreIndex: An instance of VectorStoreIndex configured with the specified Qdrant client and vector store.
    """
    if from_where=="vector_store":
        index = VectorStoreIndex.from_vector_store(embed_model=embed_model, **kwargs)
        return index
    elif from_where=="docs":
        index = VectorStoreIndex.from_documents(embed_model=embed_model, **kwargs)
        return index
    else:
        raise ValueError(f"Invalid option: {from_where}. Pick one of 'vector_store', or 'docs'.")


def create_query_pipeline(chain, verbose=True):
    """
    Creates and returns a QueryPipeline instance configured with the specified chain of components.

    Parameters:
    - chain (list): A list of components to be used in the pipeline. Each component in the list should be an instance of a module that can be used in a QueryPipeline (e.g., LLMs, query engines).
    - verbose (bool): If True, enables verbose output for the pipeline.

    Returns:
    - QueryPipeline: An instance of QueryPipeline configured with the specified chain of components.
    """
    pipeline = QueryPipeline(
        chain=chain,
        verbose=verbose
    )

    return pipeline

def create_query_engine(index, mode, **kwargs):
    """
    Creates and returns a query engine from the given index with the specified configurations.

    Parameters:
    - index: The index object from which to create the query engine. This should be an instance of VectorStoreIndex or similar, which has the as_query_engine method.
    - mode (str): The mode of the query engine to create. Possible values are "chat", "query", and "retrieve".
    - **kwargs: Additional keyword arguments for configuring the query engine, such as similarity_top_k and return_sources.

    Returns:
    - A query engine configured with the specified parameters.
    """
    if mode =="chat":
        return index.as_chat_engine(**kwargs)

    elif mode == "query":
        return index.as_query_engine(**kwargs)

    elif mode == "retrieve":
        return index.as_retriever(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}. Pick one of 'chat', 'query', or 'retrieve'.")