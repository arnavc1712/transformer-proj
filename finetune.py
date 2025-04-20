"""
    This deals with finetuning via LORA

    1. Load the GPT2 Model architecture and intialize it with the pretrained weights
    2. Create a DataLoader for the dataset
    3. [Optional] Figure out how to mask the instruction tokens so we can backpropagate only on the output tokens (jsut set the label to -100)
    4. Create a LORA module for the attention layers
    5. Evaluate 2 Scenarios:
        - Finetune the model with LORA
        - Finetune the model without LORA

    6. Compare the results using the following metrics:
        - Negative Log Likelihood
        - Perplexity
        - BLEU Score
        - ROUGE Score
    
    7. Save the model weights and the metrics in a file
"""

