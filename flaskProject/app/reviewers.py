# reviewers.py
import os
from pathlib import Path
from app.chunking.GetCode import GitLabCodeChunker
from app.embeddings import CodeEmbeddingProcessor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI

from tqdm import tqdm
import time


def getCodeReview(url, token, projectId, branch, commits):
    # 0. DB 초기화
    vectorDB = CodeEmbeddingProcessor()

    # 1. git Clone
    chunker = GitLabCodeChunker(
        gitlab_url=url,
        gitlab_token=token,
        project_id=projectId,
        local_path='./cloneRepo/' + projectId,
        branch=branch
    )
    try:
        # 2. 파일별 임베딩
        project_path = chunker.clone_project()
        if not project_path:
            return ''

        # 3. 리뷰 할 코드들 메서드 Chunking
        file_chunks = []
        for root, _, files in os.walk(project_path):
            for file in files:
                file_path = Path(root) / file

                # 불필요한 파일/폴더 제외
                if ('.git' in str(file_path)
                        or any(part.startswith('.') for part in file_path.parts)
                        or any(part.startswith('node_modules') for part in file_path.parts)):
                    continue

                # 파일 언어 확인
                language = chunker.get_file_language(str(file_path))
                if not language:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = chunker.chunk_file(str(file_path), language)
                    file_chunks.extend(chunks)

        print("청크화 끝")
        for chunk in tqdm(file_chunks):
            vectorDB.store_embeddings(chunk)


        # 4. commits 에서 코드 분리해 Chunk 화
        review_queries = [] # path, 코드, 참고할 코드
        for commit in commits:
            code_chunks = chunker.chunk_code(commit[1])
            for code_chunk in code_chunks:
                review_queries.extend([commit[0], code_chunks, vectorDB.store_embeddings(code_chunk)])

        # 5. 메서드 별 관련 코드 가져와 리트리버 생성, 질의
        openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기


        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        result = get_code_review(review_queries, llm)

        # 6. LLM 에 질의해 결과 반환
        return result

    except Exception as e:
        # 예상치 못한 오류 발생 시
        return ''


def get_code_review(review_queries, llm):
    # 메모리 초기화
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True  # 디버깅을 위해 상세 출력 활성화
    )

    # 각 코드 청크별 리뷰 수행
    for file_path, code_chunk, similar_codes in review_queries:
        # 상세 코드 리뷰 요청
        conversation.predict(input=f"""다음 코드에 대한 상세 코드 리뷰를 수행해주세요:

            파일 경로: {file_path}

            검토할 코드:
            ```
            {code_chunk}
            ```

            참고할 유사 코드들:
            ```
            {similar_codes}
            ```

            다음 관점에서 분석해주세요:
            1. 코드 품질 (가독성, 유지보수성)
            2. 잠재적인 버그나 에러
            3. 성능 개선 포인트
            4. 보안 취약점
            5. 유사 코드와 비교했을 때 개선할 점
            """)

    # 이전 대화 내용을 바탕으로 최종 요약 요청
    final_review = conversation.predict(input="""지금까지 검토한 모든 코드들의 리뷰 내용을 바탕으로 
        핵심적인 개선사항만 요약해서 제시해주세요.

        다음 형식으로 작성해주세요:
        1. 주요 개선사항 (우선순위 순)
        2. 반복적으로 발견되는 패턴
        3. 전반적인 코드 품질 평가
        4. 즉시 수정이 필요한 중요 이슈
        """)

    return {
        "summary_review": final_review,
        "conversation_memory": memory.chat_memory  # 필요시 전체 대화 내용 접근 가능
    }

# def generate_code_review(code_snippet, project_id):
#     vectordb = Chroma(
#         embedding_function=graphcodebert_embeddings,
#         collection_name=f'code_embeddings_{project_id}'
#     )
#     try:
#         # 유사한 코드 검색
#         related_codes = query_similar_code(code_snippet, project_id)
#         print("Related Codes:", related_codes)
#
#         # 문서 생성
#         if not related_codes:
#             print("No related codes found. Generating review for the provided code snippet only.")
#             documents = [Document(page_content=code_snippet)]
#         else:
#             # 입력 코드 스니펫을 포함하여 문서 리스트 생성
#             documents = [Document(page_content=code_snippet)]
#             documents.extend([Document(page_content=doc) for doc in related_codes if isinstance(doc, str)])
#
#         print("Documents:", documents)
#
#         # OpenAI API 키 설정
#         openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기
#
#
#         llm = ChatOpenAI(
#             model="gpt-4o-mini",  # 원하는 모델을 명시적으로 지정합니다
#             temperature=0,
#             openai_api_key=openai_api_key
#         )
#
#         # Retriever 생성 (Chroma의 as_retriever 사용)
#         retriever = vectordb.as_retriever()
#
#         # 커스텀 프롬프트 템플릿 정의
#         prompt_template = """
#         너는 코드리뷰 전문가야
#
#        리뷰할 메서드 코드:
#        {question}
#
#         동일 프로젝트 내 유사한 메서드들 (GraphCodeBERT 유사도 기반):
#         {context}
#
#         한국어로 리뷰 진행할 코드에 대해 유사한 동일 프로젝트의 코드를 참고해서 다음 사항을 포함하여 코드 리뷰를 작성해줘:
#         1. **코드의 기능 설명**: 코드가 수행하는 주요 기능을 간단히 설명해주세요.
#         2. **잘한 점**: 코드 작성에 있어서 효율적이거나 개선되지 않아도 좋은 부분에 대해 설명해주세요.
#         3. **개선할 부분**: 코드 품질을 높이기 위해 수정하거나 보완할 필요가 있는 부분에 대해 설명해주세요. 성능, 보안, 코드 가독성, 확장성 등을 고려하여 작성해주세요.
#
#         각 항목에 대해 유사 메서드들의 구현을 참고하여 구체적인 예시와 함께 설명해주세요, 프로젝트의 일관성을 해치지 않는 선에서 제안해주세요.
#         이 때 실제 코드는 지양하고 컨벤션에 대한 여부도 참고 코드와 유사한지
#         """
#
#         PROMPT = PromptTemplate(
#             template=prompt_template, input_variables=["context", "question"]
#         )
#
#         # context로 사용할 관련 코드 문서 내용 준비
#         context = "\n\n".join([doc.page_content for doc in documents])
#
#         # 최종 프롬프트 생성 및 출력
#         formatted_prompt = PROMPT.format(context=context, question=code_snippet)
#         print("Final Prompt to LLM:\n", formatted_prompt)
#
#         # RetrievalQA 체인 생성 (커스텀 프롬프트 사용)
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=retriever,
#             chain_type="stuff",
#             return_source_documents=False,
#             chain_type_kwargs={"prompt": PROMPT}
#         )
#
#
#
#         # 코드 리뷰 생성
#         response = qa_chain.invoke(code_snippet)
#         return response
#
#     except Exception as e:
#         print(f"Error generating code review: {e}")
#         return None
