# embeddings.py
from langchain_chroma import Chroma
from .models.codebert_model import get_code_embedding
from langchain.embeddings.base import Embeddings

# 래퍼 클래스 생성
class GraphCodeBERTEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, code_snippet):
        return get_code_embedding(code_snippet)

class CodeEmbeddingProcessor:
    def __init__(self):
        self.db = Chroma(
            embedding_function=GraphCodeBERTEmbeddings(),
            collection_name=f'code_embeddings',
            persist_directory=None  # 메모리에만 저장
        )

    # Chunk 코드 임베딩
    def store_embeddings(self, code_snippets):
        try:
            self.db.add_texts(
                texts=code_snippets,
            )
            return True
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False

    # 유사 코드 검색
    def query_similar_code(self, code_snippet, n_results=5):
        try:
            results = self.db.similarity_search_with_score(
                query=code_snippet,
                k=n_results
            )
            related_codes = [doc.page_content for doc, score in results]
            return related_codes
        except Exception as e:
            print(f"Error querying similar code: {e}")
            return []
