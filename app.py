# app.py
import streamlit as st
import os
from langgraph.graph import StateGraph, END
from playwright.async_api import async_playwright
import requests
import asyncio

# --- CONFIGURATION ---
GOOGLE_AI_API_KEY = st.secrets["GOOGLE_AI_API_KEY"]  # Store this in Streamlit Cloud's secrets
AI_STUDIO_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

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

# --- LLM NODE USING GOOGLE AI STUDIO (GEMINI) ---
def ask_llm(question):
    headers = {"Authorization": f"Bearer {GOOGLE_AI_API_KEY}"}
    data = {
        "contents": [{
            "parts": [{"text": question}]
        }],
        "generationConfig": {"maxOutputTokens": 512}
    }
    response = requests.post(AI_STUDIO_URL, headers=headers, json=data)
    try:
        result = response.json()["candidates"]["content"]["parts"]["text"]
    except Exception:
        result = "Error or unexpected response from API"
    return result

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
