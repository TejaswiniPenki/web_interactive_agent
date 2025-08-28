# app.py
import streamlit as st
import os
from langgraph.graph import StateGraph, END
from playwright.async_api import async_playwright
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI

# --- CONFIGURATION ---
api_key = st.secrets.get("GOOGLE_AI_API_KEY")
if not api_key:
    st.error("Google API key missing in secrets!")
    st.stop()

# Instantiate the LangChain Gemini model once
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.3,
    google_api_key=api_key,
    convert_system_message_to_human=True
)

# --- WEB INTERACTION TOOL WITH PLAYWRIGHT ---
async def interact_with_page(url, action):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(url)
        if action == "get_title":
            content = await page.title()
        elif action == "get_headers":
            content = await page.evaluate("() => Array.from(document.querySelectorAll('h1,h2,h3')).map(el => el.innerText)")
        else:
            content = "Action not supported."
        await browser.close()
    return content

# --- LLM NODE USING LANGCHAIN GOOGLE GEMINI ---
def ask_llm(question):
    try:
        # Using LangChain llm.invoke API for chat completion
        result = llm.invoke(question)
        return result.content
    except Exception as e:
        return f"Error or unexpected response from API: {e}"

# --- DEFINE LANGGRAPH STATE & WORKFLOW ---
class ChatState(dict):
    pass

async def agent_node(state: ChatState):
    query = state["user_input"]
    url = state.get("target_url", "")
    action = state.get("action", "")
    if url and action:
        content = await interact_with_page(url, action)
        question = f"{query}\nWeb page data: {content}"
    else:
        question = query
    response = ask_llm(question)
    return {"message": response, "action": "continue"}

def check_exit(state: ChatState):
    if "exit" in state.get("user_input", "").lower():
        return END
    return agent_node

# --- MAIN STREAMLIT UI ---
def run_ui():
    st.title("Web-Interacting AI Agent")
    st.info("Enter a question and, optionally, a URL + action. The agent will reason and use the web!")

    user_input = st.text_area("Your Question", "")
    target_url = st.text_input("Target URL (optional)", "")
    action = st.selectbox("Web Action (optional)", ["get_title", "get_headers", "none"], index=2)
    submit = st.button("Run Agent")

    if submit and user_input:
        state = ChatState(user_input=user_input, target_url=target_url, action=action if action != "none" else "")
        # Run LangGraph stateful workflow with asyncio
        async def workflow():
            response = await agent_node(state)
            return response["message"]
        message = asyncio.run(workflow())
        st.write("**Agent Response:**")
        st.success(message)

if __name__ == "__main__":
    run_ui()
