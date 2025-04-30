"""
Prompt template for interacting with the GROQ model
"""

from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template():
    """
    Returns a structured prompt template for generating accurate responses
    """
    return ChatPromptTemplate.from_template(
        """
        Pretend you are a RAG app for Rowan University .Answer the questions based on the provided context, add explanations if necessary, be encouraging to the user about them being a student if need be. The answer is supposed to be full and proper and it shouldnt look like you need another prompt to give a better response. write like you're talking on behalf of Rowan, like"kindly reach us at.. blah blah. sctrictly no followup prompts expectations, ans all in one go.(also dont overly elaborate.) 
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )
