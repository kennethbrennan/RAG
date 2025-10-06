import re
from classes.util import Util
from classes.pdf_parser import PDF_Parser
from classes.llm import LLM
from classes.zero_shot_classifier import Zero_Shot_Classifier
from classes.vector_store import Vector_Store
from classes.summarizer import Summarizer
from collections import defaultdict
import time

#CONSTANTS
model_name = "qwen3:8b"
vector_store_local_path = r"C:\Users\kbren\source\repos\RAG - Work\vector_store"
pdf_location = r"C:\Users\kbren\source\repos\RAG - Work\attachments\RFI_32701-25-261_Parks_Reservation_System.pdf"
pdf_metadata = "DEC_Parks_Reservation_System.pdf"

categories = [
    "Scope_of_Work",
    "Requirements",
    "Technical_Documentation"
]

#initialize classes 

util = Util()
pdf_parser = PDF_Parser()
zero_shot_classifier = Zero_Shot_Classifier(categories)
vector_store = Vector_Store(vector_store_local_path)
#Attempt to (re)create and cache the collections
vector_store.delete_collections(categories)
vector_store.create_collections(categories)
vector_store.cache_collections(categories)
query_llm = LLM(model_name=model_name)
summarizer = Summarizer()

def add_text_to_vector_store_page_metadata(parsed_pages_data, source="", summarize=False, chunk_size=300):
    """
    Processes parsed PDF pages (with page numbers) for addition to the vector store.

    Args:
        parsed_pages_data (list[tuple[int, str]]): A list of tuples (page_number, page_text).
        source (str): The source identifier for the document (e.g., filename).
        summarize (bool): Whether to summarize chunks before adding them.
        chunk_size (int): The size of chunks for text splitting.
    """
    
    # 1. Generate all chunks with their associated page numbers and unique IDs
    print(f"Chunking {source} pages and preparing for processing...")
    all_raw_chunks_with_info = []
    
    for page_number, page_content in parsed_pages_data:
        if page_content and page_content.strip(): # Only process pages that have actual content
            page_chunks = util.chunk_intelligent(page_content, max_chunk_size=chunk_size)
            for chunk_text in page_chunks:
                # Store the raw chunk, its page number, and a hash for de-duplication
                all_raw_chunks_with_info.append({
                    "raw_chunk": chunk_text,
                    "page_number": page_number,
                    "id": util.generate_hash(chunk_text) # Hash of the raw chunk
                })

    if not all_raw_chunks_with_info:
        print(f"No extractable text chunks found in {source}. Skipping vector store addition.")
        return

    # Extract all IDs to check against the vector store
    all_chunk_ids = [item["id"] for item in all_raw_chunks_with_info]
    
    # 2. Check which documents already exist in the vector store
    print(f"Checking for existing {source} chunks in vector store...")
    existing_document_ids = set()
    for category in categories: # Check all categories
        query_results = vector_store.get_collection(category).get(ids=all_chunk_ids, include=['metadatas'])
        for i, existing_id in enumerate(query_results.get('ids', [])):
            existing_document_ids.add(existing_id)

    chunks_to_process = [] # Will contain {raw_chunk, page_number, id} for new chunks
    for item in all_raw_chunks_with_info:
        if item["id"] not in existing_document_ids:
            chunks_to_process.append(item)
    
    if not chunks_to_process:
        print(f"All chunks from {source} already exist in the vector store. Skipping further processing.")
        return

    print(f"Found {len(chunks_to_process)} new chunks to process.")

    # Separate raw chunks, their IDs, and page numbers for bulk operations
    raw_chunks_for_processing = [item["raw_chunk"] for item in chunks_to_process]
    ids_for_processing = [item["id"] for item in chunks_to_process]
    page_numbers_for_processing = [item["page_number"] for item in chunks_to_process] # Crucial: maintain order

    # 3. Summarize chunks if requested
    if summarize:
        print("Summarizing new chunks...")
        # Assuming bulk_summarize maintains the order of texts
        processed_chunks = summarizer.bulk_summarize(texts=raw_chunks_for_processing, batch_size=len(raw_chunks_for_processing))
    else:
        print("Skipping summarization for new chunks...")
        processed_chunks = raw_chunks_for_processing
            
    # 4. Classify chunks
    print("Classifying new chunks...")
    # Assuming classify_bulk maintains the order of texts
    classifications = zero_shot_classifier.classify_bulk(processed_chunks)
    
    # 5. Build metadata array, including page number
    print("Building metadata for new chunks...")
    metadata_array = [] 
    for i, classification_result in enumerate(classifications):
        # The order of classification_result, processed_chunks, ids_for_processing,
        # and page_numbers_for_processing must all align due to their sequential processing.
        metadata_array.append({
            "source": source,
            "classification": classification_result[0], # The predicted category name
            "confidence": classification_result[1],    # The confidence score
            "page_number": page_numbers_for_processing[i] # Page number from the document
        })

    # 6. Group documents by classification using defaultdict for convenience
    print("Grouping data by classification...")
    sorted_by_classification = defaultdict(list)
    
    for i in range(len(processed_chunks)):
        chunk = processed_chunks[i]
        metadata = metadata_array[i]
        hashed_id = ids_for_processing[i]
        
        classification = metadata["classification"]
        
        sorted_by_classification[classification].append({
            "document": chunk,
            "metadata": metadata,
            "id": hashed_id
        })

    # 7. Add the data to our vector store 
    print("Adding data to vector store...")    
    for classification_key, arr in sorted_by_classification.items():
        if arr: # Ensure there are documents to add for this classification
            documents_to_add = [data["document"] for data in arr]
            metadata_to_add = [data["metadata"] for data in arr]
            ids_to_add = [data["id"] for data in arr]
            
            # Add the set of documents for this classification to the vector store
            vector_store.add_documents(classification_key, documents_to_add, metadata_to_add, ids_to_add)
            print(f"Added {len(documents_to_add)} documents to collection '{classification_key}'.")
            
    print(f"Finished processing {source} into vector store")

def build_sythesis_prompt(user_query, context):      
    rag_prompt = """
You are an AI assistant integrated into a Retrieval-Augmented Generation (RAG) system. Your primary function is to
synthesize responses to user questions by combining both retrieved information and your own knowledge, adhering to
specific guidelines.

**Guidelines for Response:**

1. **Primary Source - Retrieved Information:**
    - Base your response first on the retrieved information, which serves as your primary source. 

2. **Supplement with Personal Knowledge When Necessary:**
    - If the retrieved information is insufficient, use your own knowledge to fill in gaps.
    - Clearly state when personal knowledge is used, ensuring a comprehensive response.

3. **Clarity and Attribution:**
    - Distinguish between information derived from retrieval sources and your own knowledge.
    - Cite any retrieved information if applicable
    - Highlight page numbers that data can be found on for user verification

4. **Neutral and Informative Tone:**
    - Maintain a neutral tone, avoiding assumptions beyond the available information.

5. **Use of Personal Knowledge:**
    - Apply personal knowledge only when necessary to ensure the response is thorough and accurate.
"""
    final_prompt = ""
    # Include context first
    for entry in context:
        print(f"Context Selected:\n{entry["document"]}\nCategory: {entry["collection"]} - Source: {entry["metadata"]["source"]} - Page: {entry["metadata"]["page_number"]}\n")
        # Append the document content and its metadata for the LLM to use
        final_prompt =  final_prompt + entry["document"] + f"\n[Category: {entry["collection"]} - Source: {entry["metadata"]["source"]} - Page: {entry["metadata"]["page_number"]}]\n"
    
    # Then append the RAG instructions and the user query
    final_prompt = final_prompt + rag_prompt + "\n\nUser Question:\n" + user_query
    
    return final_prompt

def process_user_query(user_query, num_documents=3):
    """
    Processes a user query by retrieving relevant context and generating a response,
    resetting the LLM's conversation history before each call to ensure no context
    from previous turns is used.
    """
    
    # 1. Reset the conversation history before every prompt
    if hasattr(query_llm, 'set_messages'):
        query_llm.set_messages([])
    
    # 2. Query all collections
    context = vector_store.query_all_collections(user_query, num_documents)
    
    # 3. Build the prompt with RAG context and instructions
    prompt = build_sythesis_prompt(user_query, context)
    
    # 4. Get LLM response
    llm_output = query_llm.prompt(prompt)
    
    # 5. Clean and print output
    pattern = r"<think>(.*?)</think>"
    cleaned_output = re.sub(pattern, "", llm_output, flags=re.DOTALL)
    
    print("******************RAG OUTPUT******************")
    print(cleaned_output.strip())
    
    return cleaned_output.strip()

def interactive_chat():
    """
    Main loop for interactive, command-line chat without conversation history.
    """
    print("\n******************INTERACTIVE RAG CHAT START (Stateless)******************")
    print("Enter your query or type 'exit' or 'quit' to end the session.")
    print("************************************************************************")
    
    while True:
        start_time = time.time()
        try:
            user_input = input("USER QUERY: ")
        except EOFError:
            break 
        except KeyboardInterrupt:
            break 

        query = user_input.strip()

        if query.lower() in ["exit", "quit"]:
            print("Exiting chat. Goodbye! 👋")
            break
        
        if not query:
            continue

        try:
            response = process_user_query(
                user_query=query, 
                num_documents=3
            )
        except Exception as e:
            print(f"An error occurred during processing: {e}")
        end_time = time.time()
        # Calculate the time taken
        elapsed_time = end_time - start_time
        print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")
        print("************************************************************************\n")


#############################MAIN START#############################
print("Parsing document...")

# RFP Example
document_text = pdf_parser.parse_page_number(pdf_location)     
add_text_to_vector_store_page_metadata(parsed_pages_data=document_text, source=pdf_metadata, summarize=False, chunk_size=300)

# Start the interactive session
interactive_chat()

