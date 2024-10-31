from tree_sitter import Language, Parser
import tree_sitter_javascript


def get_function_nodes(node, code_bytes):  # code_bytes 매개변수 추가
    functions = []
    if node.type in [
        'function_declaration',
        'method_definition',
        'arrow_function',
        'function',
        'jsx_element',  # JSX 요소
        'jsx_fragment'  # JSX 프래그먼트
    ]:
        function_text = node_text(code_bytes, node).strip()
        if function_text and function_text != "function":
            functions.append(node)
    for child in node.children:
        functions.extend(get_function_nodes(child, code_bytes))  # code_bytes 전달
    return functions


def get_class_nodes(node):
    classes = []
    if node.type == 'class_declaration':
        classes.append(node)
    for child in node.children:
        classes.extend(get_class_nodes(child))
    return classes


def node_text(code_bytes, node):
    return code_bytes[node.start_byte:node.end_byte].decode('utf8')


def extract_functions(code_string):
    LANGUAGE = Language(tree_sitter_javascript.language())
    parser = Parser(LANGUAGE)

    code_bytes = bytes(code_string, 'utf8')  # code_bytes를 먼저 생성
    tree = parser.parse(code_bytes)
    root_node = tree.root_node
    function_nodes = get_function_nodes(root_node, code_bytes)  # code_bytes 전달

    methods = [node_text(code_bytes, node) for node in function_nodes]

    return methods