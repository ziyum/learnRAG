import re
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

st.title("AI Chatbot")

ollama_client = OpenAI(
    api_key="ollama",
    base_url="http://100.106.58.49:11434/v1",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []


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
    resp = requests.get(url, timeout=10)
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


with st.sidebar:
    st.header("⚙️ Settings")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful assistant.",
    )
    model = st.selectbox(
        "Model",
        options=["gemma4:12b"],
        index=0,
    )

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
                    st.error(f"Failed to scrape {url}")
                    print(f"Scrape error: {e}")

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

    with st.chat_message("assistant"):
        try:
            client = ollama_client
            stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": full_system_prompt},
                    *st.session_state.messages,
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        except Exception as e:
            st.error("Sorry, something went wrong. Please try again.")
            print(f"API error: {e}")
            response = "I encountered an error processing your request."
    st.session_state.messages.append({"role": "assistant", "content": response})
