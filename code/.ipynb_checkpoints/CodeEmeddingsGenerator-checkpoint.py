from typing import Generator
import json
import requests
import traceback
import os


# Parsing
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Tree, Node

# Code embedding
import torch
from unixcoder import UniXcoder


def __traverse_tree(tree: Tree) -> Generator[Node, None, None]:
    cursor = tree.walk()

    visited_children = False
    while True:
        if not visited_children:
            yield cursor.node
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break
    

def generate_code_embeddings(source_code:str, model:UniXcoder, device:str = None) -> dict:
    '''
    Uses the pre-trained model to generate code embeddings from the given source code.
    Gives embeddings for whole classes, whole methods, and tokens.
    
    Args:
        source_code: Code from which the embedding is generated.
        model: CodeBERT transformer model.
        device: PyTorch device.
    
    Returns:
        results: A dictionary of unique IDs paired with a tuple of the code embedding, 
        the type of code (class, method, or token), and the string indices for parsing out the 
        embedded code from the source code.
    '''
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {} # Store data here
    id = 0 # Each embedding will have its own ID

    # Run through every node in the parse tree
    parser = Parser(Language(tsjava.language()))
    tree = parser.parse(bytes(source_code, 'utf8'))
    for node in __traverse_tree(tree):        
        # Store results
        if node.type == 'class_declaration':
            level = 'class'
        elif node.type == 'method_declaration':
            level = 'method'
        elif (len(node.children) == 0) and (node.type != 'block_comment' and node.type != 'line_comment'): # Leaf node that is not a comment
            level = 'token'
        else: # If node does not fit one of the above levels, move on to the next
            continue
        
        # Get code embedding
        indices = (node.start_byte, node.end_byte) # Start and end indices to parse code fragments out of the source code
        code = source_code[indices[0] : indices[1]]
        with torch.no_grad():
            torch.cuda.empty_cache()
            tokens_ids = model.tokenize([code])
            source_ids = torch.tensor(tokens_ids).to(device)
            tokens_embeddings, code_embedding = model(source_ids)
            torch.cuda.empty_cache()

        results[id] = (code_embedding.tolist(), level, indices) # Add code embedding and related information
        id += 1 # ID for next embedding
    return results


def embed_all_files(files:dict) -> None:
    '''
    Generates Java code embeddings with the UniXcoder model for every given source code file.
    Results are saved in the data folder as a JSON file. 

    Args:
        files: Download URLs of the Java source code, with uniquely identifying IDs as the keys 
        (which are used in the filename of the saved JSON file).
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(device)
    # Run through every file
    for id, url in files.items():
        print(f'Generating code embeddings with #{id}: {url}')
        try:
            response = requests.get(url) # Get file
            results = generate_code_embeddings(response.text, model, device)
            json.dump(results, open(f'data/embed_{id}.json', 'w')) # Save as json
        except Exception:
            print(f'Something has gone wrong with file {id}: {url}')
            traceback.print_exc()
    return


if __name__ == "__main__":
    os.makedirs('data', exist_ok=True)
    java_files = json.load(open("download_urls.json"))
    embed_all_files(java_files)
