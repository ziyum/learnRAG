import json
import re
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import tiktoken

st.title("Customizable Chatbot")

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://100.106.58.49:11434/v1",
)

openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
deepseek_api_key = st.secrets.get("DEEPSEEK_API_KEY", "")

openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None

def is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def is_deepseek_model(model_name: str) -> bool:
    return model_name.startswith("deepseek-")


def is_ollama_model(model_name: str) -> bool:
    return model_name == "gemma4:12b"


def normalize_usage(usage):
    if not usage or not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)),
    }


def extract_content_from_response(result):
    if isinstance(result, dict):
        choice = result.get("choices", [{}])[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        return (message or {}).get("content", "") or str(choice)
    if hasattr(result, "choices"):
        choice = result.choices[0]
        if hasattr(choice, "message"):
            return getattr(choice.message, "content", "")
        return getattr(choice, "text", "") or str(choice)
    return str(result)


def get_response_usage(result):
    if isinstance(result, dict):
        return normalize_usage(result.get("usage", {}))
    if hasattr(result, "usage"):
        usage = getattr(result, "usage")
        if isinstance(usage, dict):
            return normalize_usage(usage)
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def update_token_usage(provider, usage, estimated):
    usage = normalize_usage(usage)
    total = usage["total_tokens"] or (usage["prompt_tokens"] + usage["completion_tokens"])
    st.session_state.token_usage["last_request"] = usage["prompt_tokens"] or estimated
    st.session_state.token_usage["last_response"] = usage["completion_tokens"]
    st.session_state.token_usage["last_total"] = total
    st.session_state.token_usage["total_by_provider"][provider] = (
        st.session_state.token_usage["total_by_provider"].get(provider, 0) + total
    )
    st.session_state.token_usage["estimated_total"] += estimated


def call_deepseek_chat(model, messages):
    endpoint = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {deepseek_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
    }
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict) or "choices" not in data or not data["choices"]:
        raise ValueError("Deepseek returned an unexpected response")
    content = data["choices"][0].get("message", {}).get("content", "")
    return content, normalize_usage(data.get("usage", {}))

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {
        "last_request": 0,
        "last_response": 0,
        "last_total": 0,
        "total_by_provider": {"openai": 0, "deepseek": 0, "ollama": 0},
        "estimated_total": 0,
    }


def extract_sections(soup):
    """Extract heading-anchored sections from page."""
    sections = []
    container = soup.find(["article", "main"]) or soup.body
    if not container:
        return sections

    current_heading = "Overview"
    current_paras = []

    for el in container.find_all(["h1", "h2", "h3", "p", "li"]):
        tag = el.name
        text = el.get_text(strip=True)
        if not text:
            continue
        if tag in ("h1", "h2", "h3"):
            if current_paras:
                sections.append({"heading": current_heading, "text": " ".join(current_paras)})
            current_heading = text
            current_paras = []
        else:
            current_paras.append(text)

    if current_paras:
        sections.append({"heading": current_heading, "text": " ".join(current_paras)})
    return sections


def chunk_section(text, heading, max_words=250, overlap_words=30):
    """Split a section's text into overlapping chunks, each tagged with the heading."""
    words = text.split()
    chunks = []
    stride = max_words - overlap_words
    for start in range(0, max(len(words), 1), stride):
        piece = words[start:start + max_words]
        chunks.append({
            "heading": heading,
            "text": " ".join(piece),
            "word_offset": start,
        })
        if start + max_words >= len(words):
            break
    return chunks


def scrape_url(url):
    """Scrape a URL and return heading-aware semantic chunks."""
    cleaned_url = url.strip()
    if not cleaned_url.lower().startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/121.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    resp = requests.get(cleaned_url, timeout=10, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    sections = extract_sections(soup)
    chunks = []
    for sec in sections:
        chunks.extend(chunk_section(sec["text"], sec["heading"]))
    return chunks if chunks else [{"heading": "Overview", "text": soup.get_text(separator=" ", strip=True)[:3000], "word_offset": 0}]


def retrieve_relevant(query, chunks, max_words=3000):
    """Score chunks by word overlap with the query and return top matches within budget."""
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return chunks[:5]

    scored = []
    for c in chunks:
        chunk_tokens = set(re.findall(r"\w+", c["text"].lower()))
        if not chunk_tokens:
            continue
        overlap = len(query_tokens & chunk_tokens)
        score = overlap / (len(chunk_tokens) ** 0.5 + 1)
        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    selected = []
    total_words = 0
    for score, chunk in scored:
        wc = len(chunk["text"].split())
        if total_words + wc > max_words:
            continue
        selected.append(chunk)
        total_words += wc
    return selected if selected else chunks[:3]


def count_tokens_for_messages(messages, model_name):
    """Estimate token usage for a list of role/content messages."""
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except Exception:
        encoder = tiktoken.get_encoding("cl100k_base")

    total = 0
    is_openai_chat = model_name.startswith("gpt-")
    for message in messages:
        total += len(encoder.encode(message.get("content", "")))
        if is_openai_chat:
            total += 4
            if message.get("name"):
                total += -1
    if is_openai_chat:
        total += 3
    return total


if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

with st.sidebar:
    st.header("⚙️ Settings")

    with st.form("system_prompt_form"):
        system_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            key="system_prompt_text",
        )
        submitted = st.form_submit_button("Submit")

    if submitted:
        st.session_state.system_prompt = system_prompt
        st.success("System prompt updated.")

    system_prompt = st.session_state.system_prompt

    model = st.selectbox(
        "Model",
        options=[
            "gemma4:12b",
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-3.5-turbo",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
        ],
        index=0,
    )

    if is_openai_model(model) and not openai_api_key:
        st.warning("OpenAI API key missing in .streamlit/secrets.toml")
    if is_deepseek_model(model) and not deepseek_api_key:
        st.warning("Deepseek API key missing in .streamlit/secrets.toml")

    estimated_tokens = count_tokens_for_messages(
        [{"role": "system", "content": system_prompt}] + st.session_state.messages,
        model,
    )

    cols = st.columns(2)
    cols[0].metric("Estimated conversation tokens", estimated_tokens)
    cols[1].metric("Last API total tokens", st.session_state.token_usage["last_total"])
    st.metric("Last request tokens", st.session_state.token_usage["last_request"])
    st.metric("Last response tokens", st.session_state.token_usage["last_response"])
    st.caption(
        "API totals: OpenAI={} | Deepseek={} | Ollama={}".format(
            st.session_state.token_usage["total_by_provider"]["openai"],
            st.session_state.token_usage["total_by_provider"]["deepseek"],
            st.session_state.token_usage["total_by_provider"]["ollama"],
        )
    )
    st.caption("Default model is Ollama gemma4:12b. Deepseek and OpenAI are paid via their respective APIs.")

    st.divider()
    st.header("📚 RAG Sources")
    url = st.text_input("Enter a URL to scrape", key="rag_url")
    if st.button("Add URL") and url:
        if any(s["url"] == url for s in st.session_state.rag_chunks):
            st.info("URL already added.")
        else:
            with st.spinner("Scraping and chunking..."):
                try:
                    chunks = scrape_url(url)
                    st.session_state.rag_chunks.append({"url": url, "chunks": chunks})
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to scrape {url}: {e}")
                    print(f"Scrape error for {url}: {e}")

    for i, source in enumerate(st.session_state.rag_chunks):
        cols = st.columns([4, 1])
        with cols[0]:
            short = source["url"][:40] + "..." if len(source["url"]) > 40 else source["url"]
            st.caption(f"{short} ({len(source['chunks'])} chunks)")
        with cols[1]:
            if st.button("✕", key=f"remove_rag_{i}"):
                st.session_state.rag_chunks.pop(i)
                st.rerun()

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.rag_chunks:
        all_chunks = [c for s in st.session_state.rag_chunks for c in s["chunks"]]
        top_chunks = retrieve_relevant(prompt, all_chunks)
        rag_lines = ["Use the following information to answer the user's questions:\n"]
        for c in top_chunks:
            rag_lines.append(f"[{c['heading']}]\n{c['text']}\n")
        rag_context = "\n".join(rag_lines)
    else:
        rag_context = ""

    full_system_prompt = f"{system_prompt}\n\n{rag_context}" if rag_context else system_prompt
    messages = [
        {"role": "system", "content": full_system_prompt},
        *st.session_state.messages,
    ]

    with st.chat_message("assistant"):
        response = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            if is_ollama_model(model):
                client = ollama_client
                result = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )
                response = extract_content_from_response(result)
                usage = get_response_usage(result)
                st.markdown(response)
            elif is_openai_model(model):
                if not openai_client:
                    raise ValueError("OpenAI API key is not configured.")
                result = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False,
                )
                response = extract_content_from_response(result)
                usage = get_response_usage(result)
                st.markdown(response)
            else:
                if not deepseek_api_key:
                    raise ValueError("Deepseek API key is not configured.")
                response, usage = call_deepseek_chat(model, messages)
                if response:
                    st.markdown(response)
                else:
                    st.warning("Deepseek returned an empty response.")
        except Exception as e:
            st.error("Sorry, something went wrong. Please try again.")
            print(f"API error: {e}")
            response = "I encountered an error processing your request."
        finally:
            update_token_usage(
                "openai" if is_openai_model(model) else "deepseek" if is_deepseek_model(model) else "ollama",
                usage,
                count_tokens_for_messages(messages, model),
            )
    st.session_state.messages.append({"role": "assistant", "content": response})
