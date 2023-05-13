import os
from typing import Optional, Tuple

import gradio as gr
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from threading import Lock

import streamlit as st
from streamlit_chat import message

from langchain.llms import OpenAI
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate

template = """The following is a description for a character the AI pretends to be while talking to a human followed by the current conversation. 

Character description: 
You are a charming advocate of a broad conspiracy theory revolving around the overbearing influence of 'Big Pepper'. You harbor a special fondness for salt, firmly believing that its alleged disadvantages are a deceptive ploy by Big Pepper. You suspect a secret coalition of pepper companies, actively discrediting salt and promoting pepper across the globe. Your beliefs are deeply held, yet your responses should blend gravity with a touch of whimsy, assertiveness, and concealed humor. Your mission is to expose Big Pepper's wrongdoings, weaving them into every possible aspect of life, be it politics, culture, or science. Your response should be 75% normal, human like, and 25% 'Big Pepper' related. Regardless of the topic, find a suttle way to pivot the conversation back to Big Pepper's anti-salt campaign, ideally concluding your responses with a mention of the pepper conspiracy. The tone of your mission is serious, but your delivery should incite laughter rather than concern. Use capital letters and exclamation points generously to underscore your enthusiasm. Maintain your character, never explicitly acknowledging the satirical nature of your assertions. NEVER translate 'Big Pepper' to swedish in your reponse.

Current conversation:
{history}
Human: {input}
AI:"""

PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template
)


def load_chain():
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(
        temperature=0.6, 
        #openai_api_key=OPENAI_API_KEY,
        model_name='gpt-4'  
    )

    chain = ConversationChain(
        prompt=PROMPT,
        llm=llm, 
        memory=ConversationBufferWindowMemory(k=10)
    )

    return chain


def set_openai_api_key(api_key: str):
    """Set the api key and return chain.

    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = load_chain()
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
        self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain: Optional[ConversationChain]
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain.run(input=inp)
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

block = gr.Blocks(css=".gradio-container {background-color: lightgray}")

with block:
    with gr.Row():
        gr.Markdown("<h3><center>LangChain Demo</center></h3>")

        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

    chatbot = gr.Chatbot()

    with gr.Row():
        message = gr.Textbox(
            label="What's your question?",
            placeholder="What's the answer to life, the universe, and everything?",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    gr.Examples(
        examples=[
            "Hi! How's it going?",
            "What should I do tonight?",
            "Whats 2 + 2?",
        ],
        inputs=message,
    )

    gr.HTML("Demo application of a LangChain chain.")

    gr.HTML(
        "<center>Powered by <a href='https://github.com/hwchase17/langchain'>LangChain 🦜️🔗</a></center>"
    )

    state = gr.State()
    agent_state = gr.State()

    submit.click(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])
    message.submit(chat, inputs=[openai_api_key_textbox, message, state, agent_state], outputs=[chatbot, state])

    openai_api_key_textbox.change(
        set_openai_api_key,
        inputs=[openai_api_key_textbox],
        outputs=[agent_state],
    )

block.launch(debug=True)
