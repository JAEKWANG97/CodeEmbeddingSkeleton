"""
extract_functions 에 코드값 전송하면 메서드별 코드 배열로 반환함
"""

from tree_sitter import Language, Parser
import tree_sitter_java

def get_code_elements(node):
    elements = []
    # Java의 주요 코드 구조들
    if node.type in [
        'method_declaration',      # 메서드
        'class_declaration',       # 클래스
        'enum_declaration',        # 열거형
        'interface_declaration',   # 인터페이스
        'constructor_declaration', # 생성자
        'record_declaration',      # 레코드 (Java 16+)
        # 'field_declaration',       # 필드 선언
        # 'annotation_type_declaration'  # 어노테이션 타입
    ]:
        elements.append({
            'type': node.type,
            'node': node
        })
    
    for child in node.children:
        elements.extend(get_code_elements(child))
    return elements

def get_function_nodes(node):
    functions = []
    if node.type == 'method_declaration': 
        functions.append(node)
    for child in node.children:
        functions.extend(get_function_nodes(child))
    return functions

def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode('utf8')

def extract_functions(code_string):
    # Language 초기화 시 두 번째 매개변수 'java' 추가
    PY_LANGUAGE = Language(tree_sitter_java.language())  
    parser = Parser(PY_LANGUAGE)
    
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node
    function_nodes = get_function_nodes(root_node)
 
    code_bytes = bytes(code_string, 'utf8')  # 'code' -> 'code_string'
    methods = [node_text(code_bytes, node) for node in function_nodes]

    return methods

# def extract_functions(code_string):
#     PY_LANGUAGE = Language(tree_sitter_java.language())  
#     parser = Parser(PY_LANGUAGE)
    
#     tree = parser.parse(bytes(code_string, "utf8"))
#     root_node = tree.root_node
#     code_elements = get_code_elements(root_node)
    
#     code_bytes = bytes(code_string, 'utf8')
    
#     # 각 요소의 타입과 코드를 함께 반환
#     elements = [{
#         'type': element['type'],
#         'code': node_text(code_bytes, element['node'])
#     } for element in code_elements]
    
#     return elements
