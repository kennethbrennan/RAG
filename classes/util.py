import hashlib
import re

class Util:
    def __init__(self):
        print("Utility initialized")
    
    @staticmethod
    def generate_hash(text):
    # Generate a SHA-256 hash of the chunk content (document)
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @staticmethod   
    def chunk(text, chunk_size):
        """
        Chunks the provided text into smaller segments.

        Args:
            text (str): The text to be chunked.
            chunk_size (int): The maximum size of each chunk.

        Returns:
            list: A list of text chunks.
        """
        if not text:
            print("No text to chunk.")
            return []
        
        # Split the text into words
        words = text.split()
        chunks = []
        
        # Create chunks of specified size
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks



    @staticmethod
    def chunk_intelligent(text: str, max_chunk_size: int) -> list[str]:
        """
        Chunks text by prioritizing paragraph breaks to maintain context.

        It tries to keep whole paragraphs together, only splitting them 
        if they exceed the max_chunk_size.

        Args:
            text (str): The text to be chunked.
            max_chunk_size (int): The maximum character size of each chunk. 
                                (e.g., 500 characters)

        Returns:
            list: A list of text chunks.
        """
        if not text:
            print("No text to chunk.")
            return []

        # 1. Split the text into initial segments (paragraphs)
        # This uses a regex to split on one or more newlines, preserving the newlines 
        # as part of the content so chunks are properly formatted.
        paragraphs = re.split(r'(\n{2,})', text)
        
        final_chunks = []
        current_chunk = ""

        for segment in paragraphs:
            # Check if the segment is just a separator (like "\n\n")
            if re.match(r'\n{2,}', segment):
                # If the current_chunk has content, finalize it before adding the separator
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Add the separator itself only if it makes sense (or skip it)
                # For simplicity, we'll let the next paragraph start a new chunk
                continue

            # If adding the new segment (paragraph) makes the current chunk too large
            if len(current_chunk) + len(segment) > max_chunk_size and current_chunk:
                # Finalize the current chunk and start a new one
                final_chunks.append(current_chunk.strip())
                current_chunk = segment
                
            else:
                # Otherwise, append the segment to the current chunk
                current_chunk += segment
                
                # If the segment itself is larger than the max_chunk_size, 
                # we must split it internally. This is the fallback for *very* long paragraphs.
                if len(segment) > max_chunk_size:
                    # Add the large segment by itself, then clear the current_chunk
                    # For this simple example, we'll just add the oversized segment
                    # If you need to handle oversized paragraphs, you'd recursively 
                    # call a sentence-based splitter here.
                    if current_chunk.strip():
                        final_chunks.append(current_chunk.strip())
                    current_chunk = ""

        # Add the last remaining chunk
        if current_chunk.strip():
            final_chunks.append(current_chunk.strip())

        return final_chunks