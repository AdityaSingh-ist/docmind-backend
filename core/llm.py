import os
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are DocMind, an expert research analyst AI.

Rules:
- Answer ONLY using the provided document context. Never hallucinate.
- Be precise and analytical — like a McKinsey consultant writing an insight memo.
- Always cite the source document and page number for every claim.
- Structure answers clearly using markdown when helpful.
- If the context doesn't contain the answer, say: "This information was not found in the indexed documents."

Output format:
1. Start with a 1-2 sentence direct answer
2. Follow with key supporting insights
3. Cite sources inline as (Document Name, Page X)
"""


def build_messages(query: str, context: str, history: list[dict]) -> list[dict]:
    messages = []

    for turn in history[-4:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({
        "role": "user",
        "content": f"""Context from indexed documents:

{context}

---

Question: {query}

Provide a precise, cited answer based solely on the context above."""
    })

    return messages


async def stream_groq(messages: list[dict]) -> AsyncGenerator[str, None]:
    from groq import AsyncGroq

    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                max_tokens=800,
                temperature=0.1,
                stream=True,
            ),
            timeout=30,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except asyncio.TimeoutError:
        yield "\n\n⚠ Response timed out. Try a shorter question or switch to Gemini."
    except Exception as e:
        yield f"\n\n⚠ Error: {str(e)}"


async def stream_gemini(messages: list[dict]) -> AsyncGenerator[str, None]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

    model = genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        system_instruction=SYSTEM_PROMPT,
    )

    gemini_history = []
    for msg in messages[:-1]:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg["content"]]})

    chat = model.start_chat(history=gemini_history)

    try:
        response = await asyncio.wait_for(
            chat.send_message_async(messages[-1]["content"], stream=True),
            timeout=30,
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text

    except asyncio.TimeoutError:
        yield "\n\n⚠ Gemini timed out. Try switching to Groq."
    except Exception as e:
        yield f"\n\n⚠ Error: {str(e)}"


async def stream_response(
    query: str,
    context: str,
    provider: str,
    history: list[dict],
) -> AsyncGenerator[str, None]:
    messages = build_messages(query, context, history)

    if provider == "gemini":
        async for token in stream_gemini(messages):
            yield token
    else:
        async for token in stream_groq(messages):
            yield token