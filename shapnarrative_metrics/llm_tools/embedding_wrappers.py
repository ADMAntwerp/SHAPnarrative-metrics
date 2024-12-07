from abc import ABC, abstractmethod

import replicate
from openai import OpenAI
import voyageai  
# from transformers import AutoModelForCausalLM, AutoTokenizer


class EmbedWrapper(ABC):
    @abstractmethod
    def generate_embedding(self, narrative):
        """
        Generates an embedding for a given narrative, usually consisting of a handful of sentences. 

        :param prompt: The narrative to generate an embedding for.
        :return: The generated embedding vector as a numpy array
        """
        pass
    

class VoyageEmbedder(EmbedWrapper):

    def __init__(self, api_key, model="voyage-large-2-instruct"): 
        self.model=model
        self.client=vo = voyageai.Client(api_key=api_key)

    def generate_embedding(self, narrative):
        
        embedding=self.client.embed(narrative, model=self.model).embeddings[0]

        return embedding