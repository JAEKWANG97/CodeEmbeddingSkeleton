from .models.codebert_model import get_code_embedding
import chromadb
from chromadb.utils import embedding_functions
import uuid

# ChromaDB 클라이언트 및 컬렉션 초기화
client = chromadb.Client()
collection = client.create_collection(name='code_embeddings')


def store_embeddings(code_snippets, project_id):
    # 코드를 벡터화하여 ChromaDB에 저장하는 함수
    try:

        collection.add(
            documents=[code for code in code_snippets],
            embeddings=[get_code_embedding(code) for code in code_snippets],
            ids=[str(uuid.uuid4()) for _ in range(len(code_snippets))],  # 코드 스니펫마다 고유 ID
            metadatas=[{"project_id": project_id} for _ in range(len(code_snippets))]  # 코드 스니펫마다 고유 메타데이터
        )
        return True
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False


def query_similar_code(code_snippet, n_results=5):
    # 입력 코드 스니펫에 대한 유사한 코드를 검색하는 함수
    try:
        query_embedding = get_code_embedding(code_snippet)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents']
        )
        related_codes = results['documents'][0]
        return related_codes
    except Exception as e:
        print(f"Error querying similar code: {e}")
        return []
