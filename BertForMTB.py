from pytorch_transformers.modeling_bert import BertPreTrainedModel
from transformers import BertModel
import torch
from transformers.modeling_outputs import SequenceClassifierOutput
# from .file_utils import (
#     ModelOutput,
#     add_code_sample_docstrings,
#     add_start_docstrings,
#     add_start_docstrings_to_model_forward,
#     replace_return_docstrings,
# )

# _TOKENIZER_FOR_DOC = "BertTokenizer"
# _CONFIG_FOR_DOC = "BertConfig"



class BertForMTB(BertPreTrainedModel):
    def __init__(self, config, model_name,examples,mode):
        print(mode)
        super().__init__(config)
        # super(BertForMTB, self).__init__(config)
        self.num_labels = config.num_labels
        # config.num_labels
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier2 = nn.Linear(2*config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=1) 
        
        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="bert-base-uncased",
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs.hidden_states
        if mode == 'CLS':

            # pooled_output = outputs[1]

            # pooled_output = self.dropout(pooled_output)
            CLS_hidden_states = torch.squeeze(hidden_states[-1][:,0] , dim=1)  #squeeze sequence length dim
            CLS_hidden_states = self.dropout(CLS_hidden_states)
            logits = self.classifier(CLS_hidden_states)  #(batch size, hidden size) ->(batch size, label size)
            logits = self.softmax(logits)

        elif mode == 'pooling':
            repre_hidden = []
            for example in examples:
                maxpool1 = nn.MaxPool1d(example.span1[1]- example.span1[0] + 1)
                maxpool2 = nn.MaxPool1d(example.span2[1]- example.span2[0] + 1)
                h_e1_0 = hidden_states[-1][example.guid, example.span1[0]:example.span1[1]+1]
                h_e2_0 = hidden_states[-1][example.guid, example.span2[0]:example.span2[1]+1]
                h_e1 = maxpool1(h_e1_0)
                h_e2 = maxpool2(h_e2_0)
                hr =  torch.cat([h_e1,h_e2],dim = 0)
                hr.unsqueeze_(1)
                repre_hidden.append(hr)
            torch.cat(repre_hidden,dim=0)

            logits = self.classifier2(repre_hidden)  #(batch size, 2*hidden size) ->(batch size, label size)
            logits = self.softmax(logits)
          
        else: # start state 
            repre_hidden = []
            for example in examples:
                hi = hidden_states[-1][example.guid, example.span1[0]]
                hj_2 = hidden_states[-1][example.guid, example.span2[0]]
                hr =  torch.cat([hi,hj_2],dim = 0)
                hr.unsqueeze_(1)
                repre_hidden.append(hr)
            torch.cat(repre_hidden,dim=0)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


