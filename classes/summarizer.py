from transformers import pipeline

class Summarizer:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

    def summarize(self, text, max_length=150, min_length=30):
        #print(f"Summarizing text: {text}")
        summarized = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']  
        #print(f"\nSummarized text: {summarized}\n")
        return  summarized
    
    def bulk_summarize(self, texts, max_length=150, min_length=50, batch_size=8):
        """
        Summarizes multiple texts in bulk to optimize speed.

        Args:
            texts (list of str): A list of texts to summarize.
            max_length (int): The maximum length of each summary.
            min_length (int): The minimum length of each summary.
            batch_size (int): The number of texts to process in a batch.
        
        Returns:
            list of str: Summarized texts.
        """
        summaries = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_summaries = self.summarizer(batch, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.extend([summary['summary_text'] for summary in batch_summaries])
        return summaries