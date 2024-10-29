from tree_sitter import Language, Parser
import tree_sitter_c

def get_code_elements(node):
    elements = []
    # C의 주요 코드 구조들
    if node.type in [
        'function_definition',     # 함수 정의
        'struct_specifier',        # 구조체
        'enum_specifier',          # 열거형
        'union_specifier',         # 공용체
        # 'declaration',             # 전역 변수/상수 선언
        'macro_definition'         # 매크로 정의
    ]:
        elements.append({
            'type': node.type,
            'node': node
        })
    
    for child in node.children:
        elements.extend(get_code_elements(child))
    return elements

def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode('utf8')

def extract_code_elements(code_string):
    C_LANGUAGE = Language(tree_sitter_c.language())
    parser = Parser(C_LANGUAGE)
    
    tree = parser.parse(bytes(code_string, "utf8"))
    root_node = tree.root_node
    code_elements = get_code_elements(root_node)
    
    code_bytes = bytes(code_string, 'utf8')
    
    elements = [{
        'type': element['type'],
        'code': node_text(code_bytes, element['node'])
    } for element in code_elements]
    
    return elements

