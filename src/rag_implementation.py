import os
import json
import shutil
import textwrap
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import uuid
import regex as re
import torch
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

hf_logging.set_verbosity_error()

def get_console_width(max_width=100):
    return min(shutil.get_terminal_size(fallback=(100, 20)).columns, max_width)

def print_section_title(title):
    width = get_console_width()
    print("\n" + "═" * width)
    print(title.center(width))
    print("═" * width)

def print_text_block(title, content, indent=2):
    width = get_console_width()
    wrapped_text = textwrap.fill(
        str(content),
        width=max(20, width - indent * 2),
        initial_indent=" " * indent,
        subsequent_indent=" " * indent
    )
    print(f"\n{title}")
    print("─" * width)
    print(wrapped_text)

def print_context_blocks(context_items):
    width = get_console_width()
    print("\nRetrieved Context")
    print("─" * width)

    for index, item in enumerate(context_items, start=1):
        metadata = item.get("metadata", {})
        file_name = metadata.get("file_name", "unknown")
        similarity = item.get("score")
        score_text = f" | score: {similarity:.4f}" if similarity is not None else ""

        print(f"\n[{index}] source: {file_name}{score_text}")
        print(textwrap.fill(
            item.get("text", ""),
            width=max(20, width - 4),
            initial_indent="  ",
            subsequent_indent="  "
        ))

def display_results(query, retrieved_context, response):
    print_section_title("RAG RESULT")
    print_text_block("User Query", query)
    print_context_blocks(retrieved_context)
    print_text_block("LLM Response", response)

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

def get_query_embedding(query, tokenizer, model):
    tokenized_query = tokenizer(query, return_tensors = 'pt', padding=True, truncation = True)
    embedding_query_tokens = model(**tokenized_query).last_hidden_state.mean(dim=1).squeeze().tolist()
    return embedding_query_tokens

def compute_cosine_similarity(chunk_embedding, query_embedding):
    chunk_embedding_norm = np.linalg.norm(chunk_embedding) 
    query_embedding_norm = np.linalg.norm(query_embedding)
    if chunk_embedding_norm == 0 or query_embedding_norm == 0:
        return 0
    else:
        return np.dot(chunk_embedding/ chunk_embedding_norm, query_embedding/ query_embedding_norm)

def retrieve_top_scores(chunk_embedding, query_embedding, top_ranks):
    scores = {}
    for document_id, chunk_contents in chunk_embedding.items():
        for chunk_id, chunk_info in chunk_contents.items():
            embedding_chunk = np.array(chunk_info)
            similarity_score = compute_cosine_similarity(embedding_chunk, query_embedding)
            scores[(document_id, chunk_id)] = similarity_score

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse = True)[:top_ranks]
    return sorted_scores

def retrieve_top_results(top_similarity_scores):
    top_results = []
    for (document_id, chunk_id), scores in top_similarity_scores:
        top_result = (document_id, chunk_id, scores)
        top_results.append(top_result)
    return top_results

def retrieve_text_top_ranks(top_similarity_results, document_data):
    relevant_texts = []
    for result in top_similarity_results:
        doc_id = result[0]
        chunk_id = result[1]
        score = result[2]
        relevant_text = document_data[doc_id][chunk_id]
        relevant_text["score"] = score
        relevant_texts.append(relevant_text)

    return relevant_texts

def response_from_llm(query, context, llm):
    total_context = []
    for item in context:
        context = item["text"]
        total_context.append(context)

    template = """
    You are an intelligent search engine. You will be provided with some retrieved context, as well as the users query.
    Your job is to understand the request, and answer based on the retrieved context.
    Here is context:
    <context>
    {context}
    </context>
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template= template) 
    chain = prompt | llm
    response = chain.invoke({"context": total_context[0], "question": query})
    return response

def rag_implementation():
    document_directory_path = "../documents"
    model_name = "BAAI/bge-small-en-v1.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chunk_size = 200
    top_ranks = 2
    paragraph_seperator = "\n\n"
    word_seperator = " " 
    documents = chunking(document_directory_path, tokenizer, chunk_size, paragraph_seperator, word_seperator)
    with open('../database/document_store.json', 'w', encoding= 'utf-8') as f:
        json.dump(documents, f, indent=4)
    
    model = AutoModel.from_pretrained(model_name)
    chunk_embedding = get_embedding(documents, tokenizer, model)
    
    query = "how many seats did rsp win?"
    query_embedding = get_query_embedding(query, tokenizer, model)

    top_similarity_scores = retrieve_top_scores(chunk_embedding, query_embedding, top_ranks)
    top_similarity_results = retrieve_top_results(top_similarity_scores)

    with open('../database/document_store.json', 'r') as f:
        document_data = json.load(f)

    retrieve_top_text = retrieve_text_top_ranks(top_similarity_results, document_data)

    # llm initialization
    llm = OllamaLLM(model="qwen2.5:latest")

    response = response_from_llm(query, retrieve_top_text, llm)
    display_results(query, retrieve_top_text, response)


if __name__ == "__main__":
    rag_implementation()

