"""מערכת RAG לטיפים בענייני ניקיון - מבוססת LlamaIndex."""
import os

from dotenv import load_dotenv
from llama_index.core import (
    PromptTemplate,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

load_dotenv()

DATA_DIR = "data"
STORAGE_DIR = "storage"
NOT_FOUND = "לא מצאתי מידע על כך במאגר."

# --- הגדרת המודלים ---
# מודל Embeddings מקומי שתומך בעברית
Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-small",
    query_instruction="query: ",   # המודל הזה דורש תחילית לשאלות
    text_instruction="passage: ",  # ותחילית למקטעים
)
# מודל השפה (LLM)
Settings.llm = Anthropic(
    model="claude-haiku-5-5",
    max_tokens=1000,
    thinking_dict={"type": "disabled"},  # בלי שלב חשיבה: עונה ישר, מהר וזול
)
# --- פרומפט: לענות רק לפי המידע שאוחזר ---
QA_PROMPT = PromptTemplate(
    "את עוזרת לענייני ניקיון. ענה בעברית, רק על סמך המידע הבא:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    f"אם התשובה לא מופיעה במידע, ענה בדיוק: {NOT_FOUND}\n"
    "שאלה: {query_str}\n"
    "תשובה:"
)


def build_index():
    """טעינת המסמכים, חלוקה ל-Chunks, יצירת Embeddings ושמירה ב-Vector Store."""
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    splitter = SentenceSplitter(chunk_size=300, chunk_overlap=100)
    index = VectorStoreIndex.from_documents(documents, transformations=[splitter])
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    print(f"נוצר אינדקס מ-{len(documents)} מסמכים")
    return index


def load_index():
    """טעינת האינדקס מהדיסק, או בנייה שלו אם עדיין לא קיים."""
    if not os.path.exists(STORAGE_DIR):
        return build_index()
    storage = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    return load_index_from_storage(storage)


def create_query_engine():
    """Retrieval + יצירת תשובה."""
    return load_index().as_query_engine(
        similarity_top_k=5,  # מספר המקטעים שמאחזרים
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],  # סינון מקטעים לא רלוונטיים
        text_qa_template=QA_PROMPT,
    )


def ask(query_engine, question):
    """מחזירה תשובה ורשימת מקורות."""
    response = query_engine.query(question)

    # אף מקטע לא רלוונטי מספיק: השאלה מחוץ למאגר
    if not response.source_nodes:
        return NOT_FOUND, []

    answer = str(response)
    if NOT_FOUND in answer:
        return answer, []

    sources = [
        {
            "file": node.metadata["file_name"],
            "score": round(node.score, 2),
            "text": node.text,
        }
        for node in response.source_nodes
    ]
    return answer, sources


if __name__ == "__main__":
    build_index()
