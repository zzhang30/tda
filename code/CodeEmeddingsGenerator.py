import os
from pathlib import Path
from typing import Generator
import json
import requests
from traceback import print_exc

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
        

def __prepare_directory(file_url:str, sub_dir:str) -> str:
    # Create path like in the original codebase
    path = file_url[file_url.find('master')+len('master')+1 :] # Reach master branch
    if sub_dir != '':
        path = path[path.find(sub_dir) :] # Subdirectory only
    path = path[:-5] # Remove .java
    path = 'data/' + path # Put path in data folder

    # Create directory if it doesn't exist
    Path(path).mkdir(parents=True, exist_ok=True) 
    return path


def __format_class_names(classes:list[dict]) -> str:
    class_names = ['c.'+c['name'] for c in classes]
    return '_'.join(class_names)


def __format_method_names(methods:list[dict]) -> str:
    method_names = ['m.'+m['name'] for m in methods]
    return '_'.join(method_names)


def __save_as(data, path:str, extension:str, duplicate_notifs:bool = False) -> None:
    if os.path.exists(f'{path}.{extension}'):
        i = 0
        while os.path.exists(f'{path}({i}).{extension}'):
            i += 1
        final_path = f'{path}({i}).{extension}'
        if duplicate_notifs:
            print(f'A duplicate filename has been found. Saving {final_path}')
    else:
        final_path = f'{path}.{extension}'
    json.dump(data, open(final_path, 'w'))
    return


def __save_class(data, file_url:str, sub_dir:str, nested_classes:list[dict]) -> None:
    path = __prepare_directory(file_url, sub_dir)
    filename = __format_class_names(nested_classes) # Name file after the classes being nested in

    __save_as(data, path=f'{path}/{filename}', extension='json', duplicate_notifs=True)
    return


def __save_method(data, file_url:str, sub_dir:str, nested_classes:list[dict], nested_methods:list[dict]) -> None:
    path = __prepare_directory(file_url, sub_dir)

    filename = ''
    if len(nested_classes) > 0:
        filename = __format_class_names(nested_classes) # Name file after the classes being nested in
        filename += '_'
    filename += __format_method_names(nested_methods) # Name file after the methods being nested in

    __save_as(data, path=f'{path}/{filename}', extension='json', duplicate_notifs=False)
    return


def __save_token(data, file_url:str, sub_dir:str, nested_classes:list[dict], nested_methods:list[dict]) -> None:
    path = __prepare_directory(file_url, sub_dir)

    filename = ''
    if len(nested_classes) > 0:
        filename = __format_class_names(nested_classes) # Name file after the classes being nested in
        filename += '_'
    if len(nested_methods) > 0:
        filename += __format_method_names(nested_methods) # Name file after the methods being nested in
        filename += '_'
    filename += 'token' # Let's just specify that it's a token

    __save_as(data, path=f'{path}/{filename}', extension='json', duplicate_notifs=False)
    return


def __update_nesting(span, nested_classes:list[dict], nested_methods:list[dict]) -> tuple[list[dict], list[dict]]:
    # Check whether current node is nested in previous classes
    for c in nested_classes[::-1]: # Iterate backwards
        # If current code is located outside of c, remove it from the stack
        # Keep doing this until we find a class the code is in
        # Or until we find there is no class the code is in
        if (span[0] < c['span'][0]) or (span[0] > c['span'][1]):
            nested_classes.pop()
        else:
            break
    # Now with methods
    for m in nested_methods[::-1]: # Iterate backwards
        if (span[0] < m['span'][0]) or (span[0] > m['span'][1]):
            nested_methods.pop()
        else:
            break
    return nested_classes, nested_methods


def generate_code_embeddings(file_url:str, sub_dir:str, model:UniXcoder, device:str = None) -> None:
    '''
    Uses the pre-trained model to generate code embeddings from the given source code. Gives 
    embeddings for whole classes, whole methods, and tokens, and stores them in a JSON file with 
    relevant data. The data saved in a data folder, and put in a directory that mimics that of
    the original directory location of the source file.
    
    Args:
        file_url: Download URL of a source file.
        sub_dir: The subdirectory of the Github repository from which source code is taken.
        model: UniXcoder transformer model.
        device: PyTorch device.
    '''
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    source_code = requests.get(file_url).text

    # Run through every node in the parse tree
    parser = Parser(Language(tsjava.language())) # Java language parser
    tree = parser.parse(bytes(source_code, 'utf8')) # Parse tree
    nested_classes = [] # Stack for class names
    nested_methods = [] # Stack for method names
    for node in __traverse_tree(tree):
        class_name = 'foo' # Class declaration identifer (if applicable)
        method_name = 'bar' # Method declaration identifier (if applicable)

        span = (node.start_byte, node.end_byte) # Start and end indices that span the code fragments in the source code
        nested_classes, nested_methods = __update_nesting(span, nested_classes, nested_methods) # Have we exited a class or method with this next node?

        # Store results
        if node.type == 'class_declaration':
            level = 'class'

            # Find class name
            # There should be a single identifer in the children of a class declaration node
            for child in node.children:
                if child.type == 'identifier':
                    # Save class name
                    child_span = (child.start_byte, child.end_byte)
                    class_name = source_code[child_span[0] : child_span[1]]

            # Add current class
            nested_classes.append({'name':class_name, 'span':span})

        elif node.type == 'method_declaration':
            level = 'method'

            # Find method name
            # There should be a single identifer in the children of a method declaration node
            for child in node.children:
                if child.type == 'identifier':
                    # Save method name
                    child_span = (child.start_byte, child.end_byte)
                    method_name = source_code[child_span[0] : child_span[1]]

            # Add current class
            nested_methods.append({'name':method_name, 'span':span})

        elif (len(node.children) == 0) and (node.type != 'block_comment' and node.type != 'line_comment'): # Leaf node that is not a comment
            level = 'token'
        else: # If node does not fit one of the above levels, move on to the next
            continue
        
        # Get code embedding
        code = source_code[span[0] : span[1]]
        with torch.no_grad():
            torch.cuda.empty_cache()
            tokens_ids = model.tokenize([code])
            source_ids = torch.tensor(tokens_ids).to(device)
            _, code_embedding = model(source_ids)
            torch.cuda.empty_cache()
        
        # Save code embedding with metadata
        data = [code_embedding.tolist(), '', span]
        # match level:
        #     case 'class': 
        #         data[1] = class_name
        #         __save_class(data, file_url, sub_dir, nested_classes)
        #     case 'method': 
        #         data[1] = method_name
        #         __save_method(data, file_url, sub_dir, nested_classes, nested_methods)
        #     case 'token': 
        #         data[1] = code
        #         __save_token(data, file_url, sub_dir, nested_classes, nested_methods)
        if level == 'class':
            data[1] = class_name
            __save_class(data, file_url, sub_dir, nested_classes)
        elif level == 'method':
            data[1] = method_name
            __save_method(data, file_url, sub_dir, nested_classes, nested_methods)
        elif level == 'token': 
            data[1] = code
            __save_token(data, file_url, sub_dir, nested_classes, nested_methods)
            
    return


def embed_all_files(files:dict, sub_dir:str) -> None:
    '''
    Generates Java code embeddings with the UniXcoder model for every given source 
    code file. It is assumed that all the files are located in the same Github 
    repository and subdirectory.

    Args:
        files: Download URLs of the Java source code, with uniquely identifying IDs 
            as the keys.
        sub_dir: The specific subdirectory to focus on. If it is an empty string, 
            have no subdirectory to focus on, and organize everything from the master 
            branch.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UniXcoder("microsoft/unixcoder-base")
    model.to(device)
    # Run through every file
    for id, url in files.items():
        print(f'Generating code embeddings with #{id}: {url}')
        try:
            generate_code_embeddings(url, sub_dir, model, device)
        except Exception:
            print(f'Something has gone wrong with file {id}: {url}')
            print_exc()
    return


if __name__ == "__main__":
    java_files = json.load(open("download_urls.json"))
    embed_all_files(java_files, 'ivy')
