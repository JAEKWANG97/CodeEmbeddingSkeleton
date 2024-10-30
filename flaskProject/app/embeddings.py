# embeddings.py
from langchain.vectorstores import Chroma
from .models.codebert_model import get_code_embedding
from langchain.embeddings.base import Embeddings


# 래퍼 클래스 생성
class GraphCodeBERTEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, code_snippet):
        return get_code_embedding(code_snippet)


graphcodebert_embeddings = GraphCodeBERTEmbeddings()



def store_embeddings(code_snippets, project_id):
    vectordb = Chroma(
        embedding_function=graphcodebert_embeddings,
        collection_name=f'code_embeddings_{project_id}'
    )
    try:
        vectordb.add_texts(
            texts=code_snippets,
        )
        print(f"Stored {len(code_snippets)} code snippets with project_id {project_id}.")
        return True
    except Exception as e:
        print(f"Error storing embeddings: {e}")
        return False

def query_similar_code(code_snippet, project_id, n_results=5):
    """입력 코드 스니펫에 대한 유사한 코드를 검색하는 함수"""
    vectordb = Chroma(
        embedding_function=graphcodebert_embeddings,
        collection_name=f'code_embeddings_{project_id}'
    )
    try:
        # 유사도 검색 수행 (필터를 사용하여 project_id로 제한)
        results = vectordb.similarity_search_with_score(
            query=code_snippet,
            k=n_results
        )

        related_codes = [doc.page_content for doc, score in results]
        # print(f"Related Codes: {related_codes}")
        return related_codes
    except Exception as e:
        print(f"Error querying similar code: {e}")
        return []
