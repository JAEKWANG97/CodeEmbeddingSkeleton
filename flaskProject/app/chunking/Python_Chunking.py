"""
extract_functions 에 코드값 전송하면 메서드별 코드 배열로 반환함
"""

from tree_sitter import Language, Parser
import tree_sitter_python

def get_function_nodes(node):
    functions = []
    if node.type == 'function_definition':
        functions.append(node)
    for child in node.children:
        functions.extend(get_function_nodes(child))
    return functions

def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode('utf8')

def extract_functions(code_string):
    
    # Tree-sitter 파서 초기화
    PY_LANGUAGE = Language(tree_sitter_python.language())
    parser = Parser(PY_LANGUAGE)
    
    tree = parser.parse(bytes(code_string, "utf8"))

    root_node = tree.root_node

    function_nodes = get_function_nodes(root_node)
 
    code_bytes = bytes(code_string, 'utf8')
    methods = [node_text(code_bytes, node) for node in function_nodes]

    return methods
