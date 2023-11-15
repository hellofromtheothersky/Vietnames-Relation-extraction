from keras.layers import Dense, Embedding, Conv1D
from tensorflow.keras.layers import Embedding, Dense, Dropout, Input, concatenate
from tensorflow.keras.layers import GlobalMaxPool1D
from keras.models import Model
from BaseModel import BaseModel

class CNN_model(BaseModel):
    def build_model(self, using=['word_emb', 'position_emb', 'gram_emb', 'sp_emb']):
        # input_ids = Input(shape=(max_len,), name='sentence')
        # input_masks = Input(shape=(max_len,), name='masks')
        # phobert_model = CustomPhoBertModel()
        # phobert_emb = phobert_model(input_ids, input_masks)
        phobert_emb=Input(shape=(self.max_len, 768,), name='sentence')

        input_e1_pos = Input(shape=(self.max_len,), name='e1_position')
        embed_e1_pos = Embedding(172,200, input_length=self.max_len, mask_zero=True)(input_e1_pos)
        input_e2_pos = Input(shape=(self.max_len,), name='e2_position')
        embed_e2_pos = Embedding(169,200, input_length=self.max_len, mask_zero=True)(input_e2_pos)

        input_grammar = Input(shape=(self.max_len,), name='grammar_relation')
        embed_grammar = Embedding(31,100, input_length=self.max_len, mask_zero=True)(input_grammar)

        input_sp = Input(shape=(self.max_len,), name='shortest_path')
        embed_sp = Embedding(107, 500, input_length=self.max_len, mask_zero=True)(input_sp)

        input_list=[]
        if 'word_emb' in using:
            input_list.append(phobert_emb)
        if 'position_emb' in using:
            input_list.extend([embed_e1_pos, embed_e2_pos])
        if 'gram_emb' in using:
            input_list.append(embed_grammar)
        if 'sp_emb' in using:
            input_list.append(embed_sp)

        # input_list.append(phobert_emb)

        visible = concatenate(input_list)
        interp = Conv1D(filters=200, kernel_size=3, activation='relu')(visible)
        interp = GlobalMaxPool1D()(interp)
        interp = Dropout(0.2)(interp)
        output = Dense(19, activation='softmax')(interp)
        self.model = Model(inputs=[phobert_emb, input_e1_pos, input_e2_pos, input_grammar, input_sp], outputs=output)


# class CustomPhoBertModel(Layer):
#     def __init__(self, **kwargs):
#         super(CustomPhoBertModel, self).__init__(**kwargs)
#         self.bert_model = phobert_model_tf


#     def call(self, input_ids, input_mask):
#         input_ids=tf.cast(input_ids, dtype="int32")
#         input_mask=tf.cast(input_mask, dtype="int32")

#         print(input_ids)
#         # with open('phobert_emb.json', 'r') as rf:
#         #     emb_from_ids=json.load(rf)
#         # key_id='_'.join(input_ids.toList)
#         # if key_id in emb_from_ids:
#         #     result=emb_from_ids[key_id]
#         # else:
#         #     result = self.bert_model(input_ids=input_ids, attention_mask=input_mask)["last_hidden_state"]
#         #     emb_from_ids[key_id]=result
#         #     with open('phobert_emb.json', 'r') as rf:
#         #         json.dump(emb_from_ids, rf)
#         result = self.bert_model(input_ids=input_ids, attention_mask=input_mask)["last_hidden_state"]
#         return result
# # Example usage

# input_ids = tf.keras.Input(shape=(128,))
# attention_mask = tf.keras.Input(shape=(128,))

# phobert_model = CustomPhoBertModel()