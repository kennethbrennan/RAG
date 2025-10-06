import torch
from transformers import pipeline

class Zero_Shot_Classifier:
    def __init__(self, classes, random_seed=42):
        """
        Initializes the zero-shot classifier with a fixed random seed for consistency.

        Args:
            classes (list): List of possible classification labels.
            random_seed (int): Random seed for reproducibility (default=42).
            device (int): Determines whether to run on CPU or GPU
        """
        # 1. Check for GPU availability using PyTorch
        if torch.cuda.is_available():
            # Use the specified device (defaulting to 'cuda:0')
            device = 0
        else:
            # Fallback to CPU
            device = -1
        self.hypothesis_template = "This text is about {}"
        self.classes = classes

        # Load the zero-shot classification model
        self.model = pipeline(
            "zero-shot-classification", 
            model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", 
            device=device
        )
        print("Classifier initialized")
    
    def classify(self, text):
        """
        Classifies the input text using zero-shot classification.

        Args:
            text (str): The text to classify.

        Returns:
            dict: Classification results with labels and confidence scores.
        """
        # Clean the input text
        text = text.strip()

        # Perform classification
        return self.model(
            text, 
            self.classes, 
            hypothesis_template=self.hypothesis_template, 
            multi_label=False
        )

    def classify_bulk(self, chunks):
        """
        Classifies multiple chunks of text at once.

        Args:
            chunks (list): A list of text chunks to classify.

        Returns:
            list: A list of tuples containing the best classification label and score for each chunk.
        """
        # Perform batch classification
        results = self.model(chunks, self.classes, hypothesis_template=self.hypothesis_template)

        classifications = []
        for output in results:
            best_class = output['labels'][0]
            best_score = output['scores'][0]
            classifications.append((best_class, best_score))

        return classifications