import pdfplumber
import os

class PDF_Parser:
    def __init__(self):
        print("PDFParser initialized")
    
    def parse(self, pdf_location):
        """
        Parses a PDF file and extracts text from it.

        Args:
            pdf_location (str): The path to the PDF file to be parsed.

        Returns:
            str: The extracted text from the PDF file.
        """
        try:
            with pdfplumber.open(pdf_location) as pdf:
                # Initialize an empty string to hold the extracted text
                full_text = ''
                
                # Iterate over each page in the PDF
                for page in pdf.pages:
                    # Extract text from the current page and add it to full_text
                    full_text += page.extract_text() 
                    
            return full_text.strip()  # Return the text without leading/trailing whitespace
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return None  # Return None if parsing fails

    def parse_page_number(self, pdf_location):
        """
        Parses a PDF file and extracts text from it, optionally with page numbers.

        Args:
            pdf_location (str): The path to the PDF file to be parsed.

        Returns:
            str: The extracted text from the PDF file,
                 with page numbers prepended to each page's content.
            Or
            list[tuple[int, str]]: A list of tuples, where each tuple contains
                                   (page_number, page_text).
            Returns None if parsing fails.
        """
        try:
            with pdfplumber.open(pdf_location) as pdf:
                # Option 1: Concatenate text with page numbers as a single string
                # full_text = ''
                # for i, page in enumerate(pdf.pages):
                #     page_number = i + 1  # Page numbers are 1-indexed
                #     page_text = page.extract_text()
                #     if page_text: # Ensure there's text to add
                #         full_text += f"\n--- Page {page_number} ---\n"
                #         full_text += page_text
                # return full_text.strip()

                # Option 2: Return a list of tuples (page_number, page_text)
                # This is generally more flexible for further processing
                pages_with_text = []
                for i, page in enumerate(pdf.pages):
                    page_number = i + 1  # Page numbers are 1-indexed
                    page_text = page.extract_text()
                    pages_with_text.append((page_number, page_text if page_text else ""))
                
                return pages_with_text

        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return None # Return None if parsing fails
    def parse_wiki(self, pdf_location):
        """
        Parses a PDF file of a Wikipedia article, removes sections like "References", "Footnotes", and "See also",
        and extracts the main text.

        Args:
            pdf_location (str): The path to the PDF file to be parsed.

        Returns:
            str: The extracted text from the PDF file without unwanted sections.
        """
        try:
            with pdfplumber.open(pdf_location) as pdf:
                # Combine all pages into a single string
                full_text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + '\n'
                
                # Keywords indicating sections to remove
                removal_keywords = ["References", "Footnotes", "See also"]
                
                # Locate the earliest occurrence of any removal keyword
                min_index = len(full_text)
                for keyword in removal_keywords:
                    index = full_text.find(keyword)
                    if index != -1 and index < min_index:
                        min_index = index
                
                # Truncate text at the earliest keyword
                if min_index < len(full_text):
                    full_text = full_text[:min_index]
            
            return full_text.strip()  # Return the cleaned text
        except Exception as e:
            print(f"Error parsing PDF: {e}")
            return None
        

    def parse_directory(self, directory_path):
        """
        Parses all PDF files in a specified directory and returns the raw text along with the source file names.

        Args:
            directory_path (str): The path to the directory containing PDF files.

        Returns:
            list: A list of tuples, where each tuple contains the filename and text of the parsed PDF file.
        """
        parsed_files = []
    
        # Loop through all files in the directory
        for filename in os.listdir(directory_path):
            # Process only PDF files
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(directory_path, filename)
                print(f"Parsing: {filename}")
            
                # Parse the PDF and get the full text
                text = self.parse_wiki(pdf_path)
            
                # If text is extracted, add the filename and text as a tuple
                if text:
                    parsed_files.append((filename, text))
    
        return parsed_files
    
    def parse_search(self, full_text):
    # Keywords indicating sections to remove
        removal_keywords = ["References", "Footnotes", "See also"]
                
        # Locate the earliest occurrence of any removal keyword
        min_index = len(full_text)
        for keyword in removal_keywords:
            index = full_text.find(keyword)
            if index != -1 and index < min_index:
                min_index = index
                
        # Truncate text at the earliest keyword
        if min_index < len(full_text):
            full_text = full_text[:min_index]
            
        return full_text.strip()  # Return the cleaned text
