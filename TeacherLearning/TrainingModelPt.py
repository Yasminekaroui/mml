import transformers
import torch
from torch import nn
import torch.nn.functional as F


class SentenceModel(nn.Module):
    def __init__(self, modelBase, embed_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(modelBase)
        self.embed_type = embed_type

    def generateSingleEmbedding(self, input, training=False):
        if training:
            self.transformer.train()
        inds, att = input
        if self.embed_type == "cls":
            single_embs = self.transformer(input_ids=inds, attention_mask=att).last_hidden_state[:,0,:]
        elif self.embed_type == "avg":
            embs= self.transformer(input_ids=inds, attention_mask=att).last_hidden_state
            sampleLength = att.sum(dim=-1, keepdims=True) 
            maskedEmbs = embs * torch.unsqueeze(att, -1)
            single_embs = maskedEmbs.sum(dim=1) / sampleLength
        elif self.embed_type == "last_hidden":
             single_embs = self.transformer(input_ids=inds, attention_mask=att).last_hidden_state
        
        return single_embs
    def save_pretrained(self, saveName):
        self.transformer.save_pretrained(saveName)

    def from_pretrained(self, saveName):
        self.transformer = transformers.AutoModel.from_pretrained(saveName)

    def forward(self, inputs, training):
        return self.generateSingleEmbedding(inputs, training)

    """
    def call(self, inputs, training=False, mask=None):
        return self.generateSingleEmbedding(inputs, training)
    """

'''
class SentenceModelWithLinearTransformation(SentenceModel):
    def __init__(self, modelBase, inputSize=768, embeddingSize=768, *args, **kwargs):
        super().__init__(modelBase, *args, **kwargs)
        # self.postTransformation = tf.keras.layers.Dense(embeddingSize, activation='linear')
        self.postTransformation = nn.Linear(inputSize, embeddingSize)
        self.double()

    def forward(self, inputs, training=False):
        x = self.generateSingleEmbedding(inputs, training)
        x = self.postTransformation(x)
        return x

    """
    def call(self, inputs, training=False, mask=None):
        return self.postTransformation(self.generateSingleEmbedding(inputs, training))
    """
'''
