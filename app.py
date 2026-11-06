import streamlit as st
import chromadb
import json
import urllib.request
import urllib.error

from sentence_transformers import SentenceTransformer

# إعدادات JAB.AI
MODEL_NAME = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

THRESHOLD = 0.65
TOP_K = 3

st.set_page_config(
    page_title="JAB.AI",
    page_icon="🤖",
    layout="centered"
)

# تصميم الواجهة العربية
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


# تحميل نموذج البحث وقاعدة المعرفة
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

    # قاعدة بيانات مؤقتة على الخادم
    client = chromadb.Client()

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


# البحث في قاعدة المعرفة
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


# توليد الإجابة باستخدام Groq
def ask_groq(question, context):

    if "GROQ_API_KEY" not in st.secrets:
        raise ValueError(
            "مفتاح Groq غير موجود في إعدادات Secrets."
        )

    api_key = st.secrets["GROQ_API_KEY"]

    system_prompt = """
أنت مساعد JAB.AI المتخصص في مدينة الخليل الفلسطينية.

التزم بالتعليمات التالية:
1. أجب باللغة العربية فقط.
2. استخدم المعلومات الموجودة في السياق فقط.
3. لا تخترع معلومات أو حقائق.
4. إذا لم تجد الإجابة في السياق، قل:
لا توجد معلومات كافية في قاعدة المعرفة للإجابة عن هذا السؤال.
5. اجعل الإجابة واضحة ومختصرة.
"""

    user_prompt = f"""
المعلومات المتاحة من قاعدة المعرفة:

{context}

سؤال المستخدم:
{question}

أجب اعتمادًا على المعلومات المتاحة فقط.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    }

    request = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(
            payload,
            ensure_ascii=False
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        return result["choices"][0]["message"]["content"].strip()

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"خطأ من Groq ({error.code}): {details[:500]}"
        )


# عنوان الموقع
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


# تجهيز النظام
with st.spinner("جاري تجهيز قاعدة المعرفة..."):

    try:
        embedder, collection = load_resources()
        ready = True

    except Exception as error:
        ready = False
        st.error(f"تعذر تجهيز النظام: {error}")


# واجهة المحادثة
if ready:

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "أهلاً بك! ❤️ أنا JAB.AI، "
                    "اسألني عن تاريخ الخليل ومعالمها وتراثها."
                )
            }
        ]

    # عرض المحادثة السابقة
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # استقبال السؤال
    question = st.chat_input(
        "اكتب سؤالك عن مدينة الخليل..."
    )

    if question:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("JAB.AI يفكر..."):

                try:

                    relevant = search_knowledge(
                        question,
                        embedder,
                        collection
                    )

                    if not relevant:

                        answer = (
                            "عذراً، لا توجد معلومات كافية "
                            "في قاعدة المعرفة للإجابة عن هذا السؤال."
                        )

                    else:

                        context = "\n\n".join(relevant)

                        answer = ask_groq(
                            question,
                            context
                        )

                    st.markdown(answer)

                    with st.expander(
                        "المعلومات التي اعتمدت عليها"
                    ):

                        if relevant:
                            for item in relevant:
                                st.write(item)
                        else:
                            st.write(
                                "لم يتم العثور على معلومات مناسبة."
                            )

                except Exception as error:

                    answer = (
                        "حدث خطأ أثناء معالجة السؤال. "
                        "يرجى المحاولة لاحقاً."
                    )

                    st.error(answer)
                    st.caption(str(error))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # بدء محادثة جديدة
    if st.button("محادثة جديدة"):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "أهلاً بك! ❤️ كيف أستطيع مساعدتك؟"
            }
        ]

        st.rerun()
