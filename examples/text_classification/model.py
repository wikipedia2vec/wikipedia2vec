import torch
import torch.nn as nn
import torch.nn.functional as F


class NABoE(nn.Module):
    def __init__(self, word_embedding, entity_embedding, num_classes, dropout_prob, use_word):
        super(NABoE, self).__init__()

        self.use_word = use_word

        self.word_embedding = nn.Embedding(word_embedding.shape[0], word_embedding.shape[1], padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(word_embedding))
        self.entity_embedding = nn.Embedding(entity_embedding.shape[0], entity_embedding.shape[1], padding_idx=0)
        self.entity_embedding.weight = nn.Parameter(torch.FloatTensor(entity_embedding))

        self.attention_layer = nn.Linear(2, 1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(word_embedding.shape[1], num_classes)

    def forward(self, word_ids, entity_ids, prior_probs):
        word_sum_vector = self.word_embedding(word_ids).sum(1)
        entity_vectors = self.entity_embedding(entity_ids)

        word_norm_vector = word_sum_vector / torch.norm(word_sum_vector, dim=1, keepdim=True).clamp(min=1e-12).detach()
        entity_norm_vectors = entity_vectors / torch.norm(entity_vectors, dim=2, keepdim=True).clamp(min=1e-12).detach()
        cosine_similarities = (word_norm_vector.unsqueeze(1) * entity_norm_vectors).sum(2, keepdim=True)

        attention_features = torch.cat((prior_probs.unsqueeze(2), cosine_similarities), 2)
        attention_logits = self.attention_layer(attention_features).squeeze(-1)
        attention_logits = attention_logits.masked_fill_(entity_ids == 0, -1e32)
        attention_weights = F.softmax(attention_logits, dim=1)

        feature_vector = (entity_vectors * attention_weights.unsqueeze(-1)).sum(1)
        if self.use_word:
            word_feature_vector = word_sum_vector / (word_ids != 0).sum(dim=1, keepdim=True).float()
            feature_vector = feature_vector + word_feature_vector

        feature_vector = self.dropout(feature_vector)
        return self.output_layer(feature_vector)
