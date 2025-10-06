from langchain_ollama import OllamaLLM
import torch

class LLM:
    def __init__(self, model_name="qwen3:8b", temperature=0.3, device="cuda:0", seed=99999):
            """
            Initializes the LLM class with a specified model and temperature,
            defaulting to 'cpu' if a CUDA-enabled GPU is not available.

            Args:
                model_name (str): The name of the LLM model to use.
                temperature (float): The temperature setting for the model.
                device (str): The preferred device (default is "cuda:0").
                seed (int): The seed for reproducibility.
            """
            
            # 1. Check for GPU availability using PyTorch
            if torch.cuda.is_available():
                # Use the specified device (defaulting to 'cuda:0')
                actual_device = device
                print(f"✅ GPU available. Initializing {model_name} on {actual_device} with temperature={temperature}")
            else:
                # Fallback to CPU
                actual_device = "cpu"
                print(f"⚠️ No GPU detected or available. Initializing {model_name} on {actual_device} with temperature={temperature}")

            # 2. Initialize the model with the determined device
            # Note: OllamaLLM is assumed to correctly handle the 'cpu' device string.
            self.model = OllamaLLM(model=model_name, device=actual_device, temperature=temperature, seed=seed)
            self.messages = []

    def set_messages(self, messages):
        """Sets the conversation history."""
        self.messages = messages

    def prompt(self, text):
        """
        Sends a prompt to the model and returns the response.

        Args:
            text (str): The input text for the model.

        Returns:
            str: The response from the model.
        """
        self.messages = self.messages + [("user", text)]
        return self.model.invoke(self.messages)
