from abc import ABC, abstractmethod
import replicate
from openai import OpenAI
from openai._types import NOT_GIVEN
import anthropic
from mistralai import Mistral

class LLMWrapper(ABC):
    @abstractmethod
    def generate_response(self, prompt, history):
        """
        Generates a response to the given prompt.

        :param prompt: The input prompt to generate a response for.
        :return: The generated response as a string.
        """
        pass

class ClaudeApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6
    ):
        
        self.system_role = system_role
        self.model = model 
        self.client = anthropic.Anthropic(api_key=api_key)
        self.messages=[]
        self.temperature=temperature
        
    def generate_response(self, prompt, history=[]):

        message = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=self.temperature,
            system=self.system_role,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return message.content[0].text

class GptApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6
    ):
        self.system_role = system_role
        self.model = model 
        self.client = OpenAI(api_key=api_key)
        self.messages=[]
        self.temperature=temperature

    def generate_response(self, prompt, history=[]):

        messages=history+[{ "role": "system", "content": self.system_role }, {"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )

        return completion.choices[0].message.content
    
class MistralApi(LLMWrapper):

    def __init__(
        self,
        api_key,
        model,
        system_role="You are a teacher, skilled at explaining complex AI decisions to general audiences.",
        temperature=0.6
    ):
        
        self.system_role = system_role
        self.model = model 
        self.client = Mistral(api_key=api_key)
        self.messages=[]
        self.temperature=temperature

    def generate_response(self, prompt, history=[]):
         
        messages=history+[{ "role": "system", "content": self.system_role }, {"role": "user", "content": prompt}]

        chat_response = self.client.chat.complete(
            model= self.model,
            messages = messages,
            temperature=self.temperature
        )
        return chat_response.choices[0].message.content
    

class LlamaAPI:
    def __init__(
        self,
        api_key,
        model="llama-3-70b-instruct",
        system_role="You are a helpful assistant.",
        temperature=0.6
    ):

        self.api_key = api_key
        self.model = model
        self.system_role = system_role

        self.client = replicate.Client(api_key)
        self.temperature=temperature

    def generate_response(self, prompt, history=[], max_tokens=1024):


        prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        
        if history!=[]:

            history_prompt=f"""Conversation History: \n"""

            for el in history:
                if el["role"]=="user":
                    history_prompt+=f"USER: {el['content']}\n"
                if el["role"]=="assistant":
                    history_prompt+=f"ASSISTANT: {el['content']}\n"

            prompt=history_prompt+f"USER: {prompt}"

        output = self.client.run(
            f"meta/meta-{self.model}",
            input={
                "prompt": prompt,
                "system_prompt": self.system_role,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "prompt_template": prompt_template
            },
        )
        response_text = ""
        for item in output:
            response_text += str(item)
        return response_text
    
 

 