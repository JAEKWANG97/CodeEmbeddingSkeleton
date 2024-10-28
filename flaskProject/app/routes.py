from flask import Blueprint, request, jsonify
from . import embeddings, reviewers

# Blueprint 생성
routes_bp = Blueprint('routes', __name__)

@routes_bp.route('/api/embed_repo', methods=['POST'])
def embed_repo():
    data = request.get_json()
    code_snippets = data.get('code_snippets', [])
    ids = data.get('ids', [])

    if not code_snippets or not ids:
        return jsonify({'status': 'fail', 'message': '코드 스니펫과 ID를 제공해야 합니다.'}), 400

    result = embeddings.store_embeddings(code_snippets, ids)
    if result:
        return jsonify({'status': 'success', 'message': '코드 임베딩이 저장되었습니다.'})
    else:
        return jsonify({'status': 'fail', 'message': '임베딩 저장 중 오류가 발생했습니다.'}), 500

@routes_bp.route('/api/code_review', methods=['POST'])
def code_review():
    data = request.get_json()
    code_snippet = data.get('code_snippet', '')

    if not code_snippet:
        return jsonify({'status': 'fail', 'message': '코드 스니펫을 제공해야 합니다.'}), 400

    review = reviewers.generate_code_review(code_snippet)
    if review:
        return jsonify({'status': 'success', 'review': review})
    else:
        return jsonify({'status': 'fail', 'message': '코드 리뷰 생성 중 오류가 발생했습니다.'}), 500
