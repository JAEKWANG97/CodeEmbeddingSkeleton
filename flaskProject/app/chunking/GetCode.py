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


class GitLabCodeChunker:
    def __init__(self, gitlab_url: str, gitlab_token: str, project_id: str, local_path: str, branch: str):
        self.gitlab_url = gitlab_url
        self.gitlab_token = gitlab_token
        self.project_id = project_id
        self.branch = branch
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

            git.Repo.clone_from(clone_url, str(self.project_path))
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

    def chunk_code(self, content: str, language: str) -> List[Dict]:
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
            print(f"청크화 실패: {e}")
            return []

    def cleanup_project_directory(self):
        try:
            if self.project_path and self.project_path.exists():
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

                # projectID 경로 찾기 (project_path의 상위 디렉토리)
                project_id_path = self.project_path.parent

                # projectID 경로와 그 아래 모든 것을 삭제
                if project_id_path.exists():
                    shutil.rmtree(project_id_path, onerror=remove_readonly)
                    print(f"projectID 디렉토리 삭제 완료: {project_id_path}")

                print(f"프로젝트 디렉토리 정리 완료")
        except Exception as e:
            print(f"디렉토리 정리 중 에러 발생: {e}")

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
