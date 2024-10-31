# reviewers.py
import os
from pathlib import Path
from app.chunking.GetCode import GitLabCodeChunker
from app.embeddings import CodeEmbeddingProcessor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_openai import ChatOpenAI

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
        vectorDB.store_embeddings(file_chunks)
        print("임베딩 끝")

        # 4. commits 에서 코드 분리해 Chunk
        file_extensions = {
            'python': ['.py'],
            'java': ['.java'],
            'javascript': ['.js', '.jsx'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp']
        }

        review_queries = [] # path, 코드, 참고할 코드
        for commit in commits:
            print('commit: ', commit)
            print(commit['code'])
            language = get_language_from_extension(commit['fileName'])

            if (language == ''):
                continue
            code_chunks = chunker.chunk_code(commit['code'], language)
            print('code_chunks: ', code_chunks)
            for code_chunk in code_chunks:
                print("리뷰할 메서드: ", code_chunk)
                similar_codes = vectorDB.query_similar_code((code_chunk))
                print("유사코드: ", similar_codes)
                review_queries.extend(review_queries.append((commit['fileName'], code_chunk, similar_codes)))

        # 5. 메서드 별 관련 코드 가져와 리트리버 생성, 질의
        openai_api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=openai_api_key
        )
        result = get_code_review(review_queries, llm)
        print("질의 끝 : " , result)
        # 6. LLM 에 질의해 결과 반환
        return result

    except Exception as e:
        # 예상치 못한 오류 발생 시
        return ''


def get_language_from_extension(file_name: str) -> str:
    extension = file_name.split('.')[-1].lower()  # 확장자 추출
    # 확장자별 언어 매핑
    language_map = {
        'py': 'python',
        'java': 'java',
        'js': 'javascript',
        'jsx': 'javascript',
        'ts': 'javascript',
        'tsx': 'javascript',
        'c': 'c',
        'cpp': 'cpp',
        'h': 'c',
        'hpp': 'cpp'
    }

    return language_map.get(extension, '')

def get_code_review(review_queries, llm):
    # 프롬프트 템플릿 정의
    print(review_queries)
    review_prompt = ChatPromptTemplate.from_messages([
        ("human", """
        너는 최고의 코드 리뷰어야 
        GraphCodeBERT 를 사용해 메서드별로 기존 프로젝트 에서 유사한 메서드를 가져와 코드 리뷰 을 보냈어

            파일 경로: {file_path}

            리뷰할 메서드:
            ```
            {code_chunk}
            ```

            기존 프로젝트의 graphCodeBERT 로 유사도를 검사해 찾은 유사 메서드:
            ```
            {similar_codes}
            ```

            다음 관점에서 분석해주세요:
            1. 코드의 기능 (구현 방법)
            2. 버그나 에러
            3. 성능 개선 포인트
            """)
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("human", """지금까지 메서드별로 검토한 모든 코드들의 리뷰 내용을 바탕으로 
            리뷰 요청한 method 별로 알려주고

            다음 형식으로 MarkDown 언어로 예쁘게 작성해주세요:
            1. 구현한 기능
            2. 주요 개선사항 (우선순위 순) 
            3. 즉시 수정이 필요한 중요 이슈
            """)
    ])

    # 체인 구성
    review_chain = (
        review_prompt
        | llm
        | StrOutputParser()
    )

    summary_chain = (
        summary_prompt
        | llm
        | StrOutputParser()
    )

    # 대화 기록 저장용 리스트
    conversation_history = []

    # 각 코드 청크별 리뷰 수행
    for file_path, code_chunk, similar_codes in review_queries:
        review_result = review_chain.invoke({
            "file_path": file_path,
            "code_chunk": code_chunk,
            "similar_codes": similar_codes
        })
        conversation_history.append(review_result)

    # 최종 요약 생성
    final_review = summary_chain.invoke({
        "chat_history": "\n\n".join(conversation_history)
    })

    return final_review

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
