import os
import json
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Tree, Node
from unixcoder import UniXcoder
import torch

# Helper function to extract all code fragments and metadata in a single pass
def extract_code_fragments(node, code):
    fragments = []
    if node.type == "class_declaration":
        fragments.append({
            "name": get_node_name(node, code),
            "code": code[node.start_byte:node.end_byte].decode('utf8'),
            "level": "class",
            "start_byte": node.start_byte,
            "end_byte": node.end_byte
        })
    elif node.type == "method_declaration":
        fragments.append({
            "name": get_node_name(node, code),
            "code": code[node.start_byte:node.end_byte].decode('utf8'),
            "level": "method",
            "start_byte": node.start_byte,
            "end_byte": node.end_byte
        })
    elif len(node.children) == 0 and (node.type != 'block_comment' and node.type != 'line_comment'):
        fragments.append({
            "name": get_node_name(node, code) or f"token_{node.start_byte}",
            "code": code[node.start_byte:node.end_byte].decode('utf8'),
            "level": "token",
            "start_byte": node.start_byte,
            "end_byte": node.end_byte
        })
    
    for child in node.children:
        fragments.extend(extract_code_fragments(child, code))
    
    return fragments

# Helper function to get the name of a node, if applicable
def get_node_name(node, code):
    for child in node.children:
        if child.type == "identifier":  # Commonly used for class/method names
            return code[child.start_byte:child.end_byte].decode('utf8')
    return None

# Function to generate embeddings with UniXcoder
def generate_embedding(model, device, code):
    tokens_ids = model.tokenize([code])
    source_ids = torch.tensor(tokens_ids).to(device)
    with torch.no_grad():
        _, code_embedding = model(source_ids)
    return code_embedding.tolist()

# Parse a single Java file and extract all fragments with embeddings
def parse_java_file(file_path, model, device):
    # Read Java file
    with open(file_path, 'rb') as f:
        java_code = f.read()

    # Parse the file
    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(java_code)
    root_node = tree.root_node

    # Extract all fragments (class, method, and token) in a single pass
    fragments = extract_code_fragments(root_node, java_code)
    embedded_fragments = []
    for fragment in fragments:
        embedding = generate_embedding(model, device, fragment["code"])
        embedded_fragments.append({
            "name": fragment["name"],
            "embedding": embedding,
            "level": fragment["level"],
            "position": {"start_byte": fragment["start_byte"], "end_byte": fragment["end_byte"]}
        })
    return embedded_fragments

# Extract and save fragments with embeddings to organized directory structure
def extract_and_save(directory):
    # Initialize UniXcoder model and device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(device)

    # Determine the parent directory of IVY
    ivy_parent_dir = os.path.dirname(directory)

    # Create directories for ivyClass, ivyMethod, and ivyToken
    ivy_class_dir = os.path.join(ivy_parent_dir, "ivyClass")
    ivy_method_dir = os.path.join(ivy_parent_dir, "ivyMethod")
    ivy_token_dir = os.path.join(ivy_parent_dir, "ivyToken")

    os.makedirs(ivy_class_dir, exist_ok=True)
    os.makedirs(ivy_method_dir, exist_ok=True)
    os.makedirs(ivy_token_dir, exist_ok=True)

    # Traverse directory to find all .java files
    for root, dirs, files in os.walk(directory):
        # Check if there are .java files in the current directory
        java_files = [file for file in files if file.endswith(".java")]
        if java_files:
            # Mirror the relative structure of the current root under each ivyX directory
            relative_path = os.path.relpath(root, directory)

            for java_file in java_files:
                java_file_path = os.path.join(root, java_file)

                # Create a folder for each .java file in the corresponding directories
                class_sub_dir = os.path.join(ivy_class_dir, relative_path, java_file[:-5])
                method_sub_dir = os.path.join(ivy_method_dir, relative_path, java_file[:-5])
                token_sub_dir = os.path.join(ivy_token_dir, relative_path, java_file[:-5])

                os.makedirs(class_sub_dir, exist_ok=True)
                os.makedirs(method_sub_dir, exist_ok=True)
                os.makedirs(token_sub_dir, exist_ok=True)

                # Parse the file once and extract all fragments with embeddings
                print(f"Parsing {java_file_path}")
                embedded_fragments = parse_java_file(java_file_path, model, device)

                # Save fragments in corresponding directories based on their level
                for fragment in embedded_fragments:
                    if fragment["level"] == "class":
                        output_dir = class_sub_dir
                    elif fragment["level"] == "method":
                        output_dir = method_sub_dir
                    elif fragment["level"] == "token":
                        output_dir = token_sub_dir
                    else:
                        continue

                    # Create a JSON file for the fragment
                    fragment_path = os.path.join(output_dir, f"{fragment['name']}.json")
                    with open(fragment_path, 'w') as json_file:
                        json.dump({
                            "embedding": fragment["embedding"],
                            "level": fragment["level"],
                            "position": fragment["position"]
                        }, json_file, indent=2)
if __name__ == "__main__":
    extract_and_save("/scratch/zzhang30/cs420/data/ivy")
