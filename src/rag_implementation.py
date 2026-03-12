import os
import json
from transformers import AutoTokenizer, AutoModel
import uuid
import regex as re
import torch

def chunking(document_directory_path, tokenizer, chunk_size, paragraph_seperator = "\n\n", word_seperator= " "):
    documents = {}
    file_names = os.listdir(document_directory_path)
    for file in file_names:
        file_path = os.path.join(document_directory_path, file)
        basename = os.path.basename(file_path)
        stock_keeping_unit = os.path.splitext(basename)[0] #get the filename without extension
        content=""
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

        except FileNotFoundError:
            print(f"Error: the file at {file_path} was not found")
        except UnicodeDecodeError:
            print(f"Error: Could not decode the file using utf-8. Try another encoding")

        document_unique_id = str(uuid.uuid4())

        file_chunks = {}

        paragraphs = re.split(pattern=paragraph_seperator, string=content)
        
        for paragraph in paragraphs:
            words = paragraph.split(word_seperator)
            
            current_word_list = []
            completed_chunk_text = ""
            
            overlap_need = 20
    
            for word in words:
                current_word_list.append(word)
                string_to_tokenize = word_seperator.join(current_word_list)
                tokenized_string = tokenizer.tokenize(string_to_tokenize)
                if len(tokenized_string) > chunk_size:
                    overflowing_word = current_word_list[-1]
                    current_word_list = current_word_list[:-1]
                    completed_chunk_text = word_seperator.join(current_word_list)
                    
                    chunk_unique_id = str(uuid.uuid4())
                    file_chunks[chunk_unique_id] = {
                            "text" : completed_chunk_text,
                            "metadata": {"file_name": stock_keeping_unit}
                    }
                      
                    overlapped_word = current_word_list[-overlap_need:]
                    current_word_list = overlapped_word + [overflowing_word]

            if current_word_list: 
                completed_chunk_text = word_seperator.join(current_word_list)
                chunk_unique_id = str(uuid.uuid4())
                file_chunks[chunk_unique_id] = {
                    "text": completed_chunk_text,
                    "metadata": {"file_name": stock_keeping_unit}
                }

        documents[document_unique_id] = file_chunks

    return documents

def get_embedding(documents, tokenizer, model):
    chunk_embeddings_document ={}
    for document_id, chunks_info in documents.items():
       chunk_embeddings_chunkinfo = {}
       for chunk_id, chunk_text in chunks_info.items():
           text = chunk_text.get("text")
           tokenized_text = tokenizer(text, return_tensors='pt', padding= True, truncation= True)
           with torch.no_grad():
               embedding_text_tokens = model(**tokenized_text).last_hidden_state.mean(dim=1).squeeze().tolist()
           
           chunk_embeddings_chunkinfo[chunk_id] = embedding_text_tokens
       chunk_embeddings_document[document_id] = chunk_embeddings_chunkinfo
    return chunk_embeddings_document

def rag_implementation():
    document_directory_path = "../documents"
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunk_size = 200
    paragraph_seperator = "\n\n"
    word_seperator = " " 
    documents = chunking(document_directory_path, tokenizer, chunk_size, paragraph_seperator, word_seperator)
    with open('../database/document_store.json', 'w', encoding= 'utf-8') as f:
        json.dump(documents, f, indent=4)
    
    model = AutoModel.from_pretrained(model_name)
    chunk_embedding = get_embedding(documents, tokenizer, model)
    print(chunk_embedding)

if __name__ == "__main__":
    rag_implementation()

