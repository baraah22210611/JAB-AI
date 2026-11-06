import streamlit as st
import chromadb
import json
import urllib.request
import urllib.error

from sentence_transformers import SentenceTransformer

MODEL_NAME = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
THRESHOLD = 0.65
TOP_K = 3

st.set_page_config(
    page_title="JAB.AI",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #f5f7fb;
}
.block-container {
    max-width: 850px;
    padding-top: 2rem;
}
h1, h2, h3, p, label {
    direction: rtl;
    text-align: right;
}
.chat-title {
    text-align: center;
    direction: rtl;
    color: #243b67;
    font-size: 38px;
    font-weight: bold;
}
.chat-subtitle {
    text-align: center;
    direction: rtl;
    color: #667085;
    font-size: 17px;
    margin-bottom: 30px;
}
.stChatMessage {
    direction: rtl;
    text-align: right;
    border-radius: 15px;
}
[data-testid="stChatInput"] {
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_resources():
    embedder = SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )

    with open("data.txt", "r", encoding="utf-8") as file:
        content = file.read()

    chunks = [
        part.strip()
        for part in content.split("\n\n")
        if part.strip()
    ]

    if not chunks:
        raise ValueError("ملف المعرفة فارغ!")

    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("hebron_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="hebron_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[f"doc_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings
    )

    return embedder, collection


def search_knowledge(question, embedder, collection):
    query_embedding = embedder.encode(
        [question],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K, collection.count()),
        include=["documents", "distances"]
    )

    documents = results["documents"][0] or []
    distances = results["distances"][0] or []

    relevant = [
        doc
        for doc, distance in zip(documents, distances)
        if distance <= THRESHOLD
    ]

    return relevant


def ask_qwen(question, context):
    prompt = f"""
أنت مساعد JAB.AI المتخصص في مدينة الخليل الفلسطينية.

التزم بالتعليمات:
1. أجب باللغة العربية.
2. استخدم المعلومات الموجودة في السياق فقط.
3. لا تخترع معلومات.
4. إذا لم تجد الإجابة في السياق، قل:
لا توجد معلومات كافية في قاعدة المعرفة للإجابة عن هذا السؤال.
5. اجعل الإجابة واضحة ومختصرة.

السياق:
{context}

السؤال:
{question}

الإجابة:
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {"temperature": 0}
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(
            response.read().decode("utf-8")
        )

    return result["message"]["content"].strip()


st.markdown(
    '<div class="chat-title">🤖 JAB.AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="chat-subtitle">'
    'مساعدك الذكي للتعرّف على مدينة الخليل'
    '</div>',
    unsafe_allow_html=True
)

with st.spinner("جاري تجهيز قاعدة المعرفة..."):
    try:
        embedder, collection = load_resources()
        ready = True
    except Exception as error:
        ready = False
        st.error(f"تعذر تجهيز النظام: {error}")

if ready:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "أهلاً بك! ❤️ اسألني عن تاريخ الخليل ومعالمها وتراثها."
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(
                message["content"],
                unsafe_allow_html=False
            )

    question = st.chat_input("اكتب سؤالك عن مدينة الخليل...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("JAB.AI يفكر..."):
                try:
                    relevant = search_knowledge(
                        question, embedder, collection
                    )

                    if not relevant:
                        answer = (
                            "عذراً، لا توجد معلومات كافية "
                            "في قاعدة المعرفة للإجابة عن هذا السؤال."
                        )
                    else:
                        context = "\n\n".join(relevant)
                        answer = ask_qwen(question, context)

                    st.markdown(answer)

                    with st.expander("المعلومات التي اعتمدت عليها"):
                        if relevant:
                            for item in relevant:
                                st.write(item)
                        else:
                            st.write("لم يتم العثور على معلومات مناسبة.")

                except Exception as error:
                    answer = f"حدث خطأ: {error}"
                    st.error(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    if st.button("محادثة جديدة"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "أهلاً بك! ❤️ كيف أستطيع مساعدتك؟"
            }
        ]
        st.rerun()