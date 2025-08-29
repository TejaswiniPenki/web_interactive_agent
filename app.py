import streamlit as st
import asyncio
import os

from playwright.async_api import async_playwright
from langchain_google_genai import ChatGoogleGenerativeAI

# Automatically install browser binaries at runtime (critical for Streamlit Cloud)
if not os.path.exists("/home/appuser/.cache/ms-playwright"):
    os.system("playwright install chromium")

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Missing Gemini API key in secrets!")
    st.stop()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

st.title("💡 Robust Web-Interacting Agent")

st.info("Paste any URL below. Optionally provide login credentials, take screenshots, and let Gemini AI summarize the content.")

url = st.text_input("Website URL", "")
username = st.text_input("Username (optional)", "")
password = st.text_input("Password (optional)", "", type="password")
take_screenshot = st.checkbox("Take Screenshot after login", value=True)
question = st.text_area("Ask Gemini AI about page (optional)", "")

submit = st.button("Run Agent")

async def automate_web(url, username, password, screenshot_path="page.png"):
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        if username and password:
            try:
                await page.fill('input[type="text"], input[name*="user"], input[name*="email"]', username)
                await page.fill('input[type="password"]', password)
                await page.click('button[type="submit"], input[type="submit"]')
                await page.wait_for_load_state("networkidle")
            except Exception as e:
                st.warning("Login fields not found or login failed. Check site selectors or credentials.")
        screenshot = None
        if take_screenshot:
            await page.screenshot(path=screenshot_path)
            screenshot = screenshot_path
        content = await page.content()
        await browser.close()
    return content, screenshot

if submit and url:
    with st.spinner("Agent working..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        page_content, screenshot_path = loop.run_until_complete(
            automate_web(url, username, password)
        )
        st.success("Web automation complete!")
        if screenshot_path:
            st.image(screenshot_path, caption="Screenshot of Page")
        if question:
            prompt = f"{question}\n\nPage HTML:\n{page_content[:5000]}..."
            ai_response = llm.invoke(prompt)
            st.markdown("### Gemini AI Output")
            st.write(ai_response.content)
        else:
            st.markdown("### Page HTML snippet:")
            st.code(page_content[:2000])

# This setup has no local dependency. Directly push repo. Streamlit Cloud will install everything.
