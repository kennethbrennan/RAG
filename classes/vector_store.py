from datetime import datetime
from typing import Collection
import chromadb
from chromadb.utils import embedding_functions
import uuid

class Vector_Store:
    def __init__(self, storage_path):
        """
        Initializes the vector store to a local copy using the default embeddings model

        Args:
            storage_path (string): A file location that the vector store will be saved in (local)
        """
        self.client = chromadb.PersistentClient(path=storage_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.cached_collections = {}
        print("Vector Store initialized")
        
    def heartbeat(self):
        """
        Checks the connection of the local client
        """
        return self.client.heartbeat()
    
    def cache_collections(self, collection_list):
        """
        Caches each collection from collection_list, for improved response times. 
        This can lead to large memory usage if there are too many collections but some bulk operations rely on this.

        Args:
            collection_list (list): A list of collection names to be cached
        """    
        for collection_name in collection_list:    
            collection = self.client.get_collection(name=collection_name)
            self.cached_collections[collection_name] = collection
            
    def create_collection(self, collection_name):
        """
        Creates a new collection or retrieves an existing one from the cache.

        Args:
            collection_name (str): The name of the collection to be created or retrieved.

        Returns:
            object: The created or cached collection object.
        """
        # Check if the collection already exists in the cache, if so return it
        cached_collection = self.cached_collections.get(collection_name)
        if cached_collection is not None:
            return cached_collection

        # If the collection does not exist, create it
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"This is the collection containing documents about {collection_name}",
                "created": str(datetime.now())
            })
        # Cache the newly created collection
        self.cached_collections[collection_name] = collection
        return collection

    def create_collections(self, collection_list):
        """
        Creates multiple collections based on the provided list of collection names.

        Args:
            collection_list (list): A list of collection names to be created.

        Returns:
            None
        """
        print("Creating collections...")
        success = []  # List to track successfully created collections

        for collection_name in collection_list:
            try:
                # Attempt to create each collection
                self.create_collection(collection_name)
                success.append(collection_name)
            except Exception as e:
                # Log the error for collections that failed to be created
                print(f"Failed to create collection {collection_name}: {e}")

        # Print a summary of successfully created collections
        print(f"Collections Created: {success}")

    
    def get_collection(self, collection_name):
        """
        Retrieves a collection by name. Checks the cache first before fetching from the client.

        Args:
            collection_name (str): The name of the collection to retrieve.

        Returns:
            object: The collection object corresponding to the specified name.
        """
        # Check the cache for the collection
        cached_collection = self.cached_collections.get(collection_name)
        if cached_collection is not None:
            # Return the cached collection if it exists
            return cached_collection

        # If the collection is not found in the cache, fetch it from the client
        collection = self.client.get_collection(name=collection_name)
        # Cache the fetched collection for future requests
        self.cached_collections[collection_name] = collection
        return collection
    
    def add_document(self, collection_name, document, metadata=None, id=None):
        """
        Adds a document to the specified collection. Uses a UUID4 as the document ID if none is provided.

        Args:
            collection_name (str): The name of the collection to which the document will be added.
            document (str): The document to be added.
            metadata (dict, optional): Metadata associated with the document. Defaults to an empty dictionary.
            id (str, optional): The unique identifier for the document. If not provided, a UUID4 will be generated.

        Returns:
            None
        """
        # Use an empty dictionary as the default metadata if none is provided
        if metadata is None:
            metadata = {}
        # Generate a UUID4 as the document ID if none is provided
        if id is None:
            id = str(uuid.uuid4())

        # Retrieve the specified collection
        collection = self.get_collection(collection_name)
        # Add the document to the collection with the associated metadata and ID
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[id]
        )

        
    #Adds a set of documents to the given collection, requires properly formated ids and sources corresponding to each document
    #While this is recommended over adding a single document at a time, it requires significantly more overhead    
    def add_documents(self, collection_name, documents, metadata, ids):
        collection = self.get_collection(collection_name)
        collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
    
    #Queries the given collection returning num_documents that match the plaintext query
    def query_collection(self, collection_name, query_text, num_documents=5):
        collection = self.get_collection(collection_name)
        return collection.query(
            query_texts=[query_text],
            n_results=num_documents
        )
    
    #Queries all collections by a list of ids, returning the ids found as a singular list
    def query_collections_by_ids(self, ids):
        document_ids = []
        for collection_name, collection in self.cached_collections.items():
            query_result = collection.get(
	            ids=ids
            )
            if query_result:
                document_ids.extend(query_result["ids"])
        return document_ids
        

    #Deletes a collection of the given name, mostly used for debugging purposes
    def delete_collection(self, collection_name):
        try:
            del self.cached_collections[collection_name]
        except Exception as e:
            print(f"Error occured while deleting collection from cache, it most likely did not exist:\n{e}")
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Error occured while deleting collection from client:\n{e}")
        print(f"Deleted Collection: {collection_name}")
            
    #Deletes all collections in the collection_list, used for resetting the state of the persistent db, mainly debugging
    def delete_collections(self, collection_list):
        for collection_name in collection_list:
            self.delete_collection(collection_name)
    
    # Queries all collections and returns the top k documents across all collections
    def query_all_collections(self, query_text, k=5):
        results = []

        for collection_name, collection in self.cached_collections.items():
            try:
                query_result = collection.query(
                    query_texts=[query_text],
                    n_results=k
                )
                for i in range(len(query_result["documents"][0])):
                    results.append({
                        "collection": collection_name,
                        "document": query_result["documents"][0][i],
                        "metadata": query_result["metadatas"][0][i],
                        "id": query_result["ids"][0][i],
                        "score": query_result["distances"][0][i]
                    })
            except Exception as e:
                print(f"Error querying collection '{collection_name}': {e}")

        # Sort results by score (ascending, as lower distance is better for similarity)
        results = sorted(results, key=lambda x: x["score"])
        return results[:k]        