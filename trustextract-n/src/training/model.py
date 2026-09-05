import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.modeling_outputs import TokenClassifierOutput
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class MultiTaskOutput(TokenClassifierOutput):
    """
    Output class for Multi-Task Model containing both token logits (NER)
    and sequence logits (Document Classification).
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    doc_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class TrustExtractMultiTaskModel(PreTrainedModel):
    """
    Custom PyTorch model that performs Multi-Task Learning on MuRIL.
    Head 1: Token Classification (NER) for extracting fields (Title, Date, Authority).
    Head 2: Sequence Classification on the [CLS] token for Document Type.
    """
    
    def __init__(self, config, num_token_labels, num_doc_labels, doc_loss_weight=0.2):
        super().__init__(config)
        self.num_token_labels = num_token_labels
        self.num_doc_labels = num_doc_labels
        self.doc_loss_weight = doc_loss_weight

        # Base encoder (e.g., MuRIL)
        self.encoder = AutoModel.from_config(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Head 1: Token Classification
        self.token_classifier = nn.Linear(config.hidden_size, num_token_labels)
        
        # Head 2: Document Type Classification (using pooled output / CLS)
        self.doc_classifier = nn.Linear(config.hidden_size, num_doc_labels)
        
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,       # Token labels
        doc_labels=None,   # Document type labels
        **kwargs
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs
        )
        
        # Sequence output for token classification
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        token_logits = self.token_classifier(sequence_output)
        
        # Pooled output (CLS token) for document classification
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        doc_logits = self.doc_classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Flatten predictions for token loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            token_loss = loss_fct(active_logits, active_labels)
            loss = token_loss
            
            # Add document loss if provided
            if doc_labels is not None:
                doc_loss = loss_fct(doc_logits, doc_labels)
                loss = token_loss + (self.doc_loss_weight * doc_loss)

        return MultiTaskOutput(
            loss=loss,
            logits=token_logits,
            doc_logits=doc_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
