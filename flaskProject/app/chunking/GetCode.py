import os
import stat
import sys
from pathlib import Path
import gitlab
import git
import json
import shutil
from typing import Dict, List, Optional

from .Python_Chunking import extract_functions as python_extract
from .Java_Chunking import extract_functions as java_extract
from .JavaScript_Chunking import extract_functions as javascript_extract
from .C_Chunking import extract_code_elements as c_extract
from app.embeddings import store_embeddings


class GitLabCodeChunker:
    def __init__(self, gitlab_url: str, gitlab_token: str, project_id: str, local_path: str):
        self.gitlab_url = gitlab_url
        self.gitlab_token = gitlab_token
        self.project_id = project_id
        self.local_path = Path(local_path)
        self.gl = gitlab.Gitlab(gitlab_url, private_token=gitlab_token)
        self.project_path = None

        # 지원하는 파일 확장자
        self.file_extensions = {
            'python': ['.py'],
            'java': ['.java'],
            'javascript': ['.js', '.jsx'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.hpp']
        }

    def clone_project(self) -> str:
        """GitLab 프로젝트를 로컬에 클론"""
        try:
            # GitLab 프로젝트 정보 가져오기
            project = self.gl.projects.get(self.project_id)

            # 클론 URL 생성
            clone_url = project.http_url_to_repo.replace('https://', f'https://oauth2:{self.gitlab_token}@')

            # 로컬 경로 생성
            self.project_path = self.local_path / project.path
            self.project_path.mkdir(parents=True, exist_ok=True)

            if not list(self.project_path.iterdir()):  # 디렉토리가 비어있는 경우에만 클론
                print(f"클론 중: {project.path}")
                git.Repo.clone_from(clone_url, str(self.project_path))
                print(f"클론 완료: {project.path}")
            else:
                print(f"이미 존재하는 프로젝트: {project.path}")

            return str(self.project_path)

        except Exception as e:
            print(f"클론 중 에러 발생: {e}")
            return None

    def get_file_language(self, file_path: str) -> Optional[str]:
        """파일 확장자를 기반으로 언어 감지"""
        ext = Path(file_path).suffix.lower()
        for lang, extensions in self.file_extensions.items():
            if ext in extensions:
                return lang
        return None

    def chunk_file(self, file_path: str, language: str) -> List[Dict]:
        """파일을 청크로 분할"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                print(f"파일 읽기 실패: {file_path} - {e}")
                return []

        try:
            if language == 'python':
                chunks = python_extract(content)
            elif language == 'java':
                chunks = java_extract(content)
            elif language == 'javascript':
                chunks = javascript_extract(content)
            elif language in ['c', 'cpp']:
                chunks = c_extract(content)
            else:
                return []

            return chunks
        except Exception as e:
            print(f"청크화 실패: {file_path} - {e}")
            return []

    def cleanup_project_directory(self):
        """클론된 프로젝트 디렉토리와 그 내부의 모든 파일 삭제"""
        try:
            if self.project_path and self.project_path.exists():
                #### chunking 디렉토리를 임시 위치로 이동 -> 이후 json 데이터를 임베딩 할꺼라 지워질 것 ####
                chunking_path = self.project_path / 'chunking'
                if chunking_path.exists():
                    temp_chunking_path = self.local_path / 'chunking'
                    if temp_chunking_path.exists():
                        shutil.rmtree(temp_chunking_path)
                    shutil.move(str(chunking_path), str(self.local_path))
                ###################################################################################

                # Git 저장소 객체 정리
                try:
                    repo = git.Repo(self.project_path)
                    repo.close()
                except:
                    pass

                import time
                time.sleep(1)

                # Windows에서 읽기 전용 속성 제거
                def remove_readonly(func, path, excinfo):
                    os.chmod(path, stat.S_IWRITE)
                    func(path)

                # 프로젝트 디렉토리의 상위 디렉토리
                project_parent = self.project_path.parent

                # 프로젝트 디렉토리 삭제
                if self.project_path.exists():
                    shutil.rmtree(self.project_path, onerror=remove_readonly)

                # local_path 아래의 모든 폴더 삭제
                for item in self.local_path.iterdir():
                    #### 이후 json 데이터를 임베딩 할꺼라 지워질 것 ####
                    if item.is_dir() and item.name != 'chunking':
                        ###################################################################################
                        shutil.rmtree(item, onerror=remove_readonly)

                print(f"프로젝트 디렉토리 삭제 완료")
        except Exception as e:
            print(f"디렉토리 정리 중 에러 발생: {e}")

    def process_project(self, projectId):
        """프로젝트 클론 및 파일 청크화 실행"""
        try:
            # 프로젝트 클론
            project_path = self.clone_project()
            if not project_path:
                return

            # chunking 폴더 생성
            chunking_path = Path(project_path) / 'chunking'
            chunking_path.mkdir(exist_ok=True)

            # 각 언어별 청크 저장을 위한 딕셔너리
            language_chunks = {lang: {} for lang in self.file_extensions.keys()}

            # 모든 파일 처리
            for root, _, files in os.walk(project_path):
                for file in files:
                    file_path = Path(root) / file

                    # git 및 chunking 폴더 제외
                    if '.git' in str(file_path) or 'chunking' in str(file_path):
                        continue

                    # 언어 감지
                    language = self.get_file_language(str(file_path))
                    if not language:
                        continue

                    print(f"처리 중: {file_path}")

                    # 파일 청크화
                    chunks = self.chunk_file(str(file_path), language)
                    if chunks:
                        # 상대 경로로 저장 -> 이후 json 데이터를 임베딩 할 로직 들어갈 위치 ####
                        # relative_path = str(file_path.relative_to(project_path))
                        # language_chunks[language][relative_path] = chunks
                        ###################################################################################

                        # for chunk in chunks:
                        store_embeddings(chunks, projectId)

            # # 각 언어별로 JSON 파일 저장
            # for language, chunks in language_chunks.items():
            #     if chunks:  # 청크가 있는 경우만 저장
            #         output_file = chunking_path / f"{language}_chunks.json"
            #         with open(output_file, 'w', encoding='utf-8') as f:
            #             json.dump(chunks, f, indent=2, ensure_ascii=False)
            #         print(f"{language} 청크 저장 완료: {output_file}")

            # 처리 완료 후 정리
            self.cleanup_project_directory()

        except Exception as e:
            print(f"프로젝트 처리 중 에러 발생: {e}")
            # 에러 발생시에도 정리 시도
            self.cleanup_project_directory()

        print("청킹은 문제없이 완료")


"""
프로젝트 클론 
    ↓
파일 순회
    ↓
언어 감지
    ↓
파일 청크화
    ↓
JSON 저장
    ↓
임시 파일 정리
"""
