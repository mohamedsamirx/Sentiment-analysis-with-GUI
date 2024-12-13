from torch import nn
from transformers import BertModel



# Define a class for your model if necessary
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        """
        Initializes the BERT-based classifier model.
        
        Args:
            bert_model_name (str): The name of the pre-trained BERT model to load (e.g., 'bert-base-uncased').
            num_classes (int): The number of output classes (e.g., 2 for binary classification).
        """
        super(BERTClassifier, self).__init__()
        
        # Load the pre-trained BERT model from Hugging Face's transformers library
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.1)  # Dropout with a probability of 0.1
        
        # Fully connected layer that maps the BERT output to the desired number of classes
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        # self.bert.config.hidden_size: The hidden size of the BERT model (e.g., 768 for BERT-base)
        # num_classes: The number of classes for classification (e.g., 2 for binary classification)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass for the BERTClassifier.
        
        Args:
            input_ids (torch.Tensor): Tensor of token ids of shape (batch_size, sequence_length).
            attention_mask (torch.Tensor): Tensor indicating the presence of actual tokens vs padding.
        
        Returns:
            torch.Tensor: Logits output by the model of shape (batch_size, num_classes).
        """
        # Pass the input_ids and attention_mask through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the pooled output from BERT (corresponding to the [CLS] token)
        pooled_output = outputs.pooler_output
        
        # Apply dropout to the pooled output to prevent overfitting
        x = self.dropout(pooled_output)
        
        # Pass the output through the fully connected layer to get logits
        logits = self.fc(x)
        
        return logits  # Return the logits, which are the raw predictions for each class
