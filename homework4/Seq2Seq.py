"""
    seq按字分词
"""
import re
import tensorflow
from keras.preprocessing.text import Tokenizer
import numpy as np
import os


# 读取文本数据
def preprocess_text():
    with open("corpus/越女剑.txt", 'r', encoding='gb18030') as f:
        text = f.read()
        text = text.replace(
            "本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", "")
        text = text.replace("本书来自www.cr173.com免费txt小说下载站", "")
        text = text.replace("----【新语丝电子文库(www.xys.org)】", "")
        text = text.replace("----〖新语丝电子文库(www.xys.org)〗", "")
        text = text.replace("Ｖｉｋｉｎｇｓ　＜ｊｏｂｊｏｂ＠ｇｄｕｐ３．ｇｄ．ｃｅｉ．ｇｏ．ｃｎ＞", "")
        text = text.replace("Last Updated: Saturday, November 16, 1996", "")
        text = text.replace("<图片>", "")
        text = text.replace("」", "")
        text = text.replace("「", "")
        text = text.replace('\u3000', '')
        text = text.replace('\n', '')
        # text = text.replace('\t', '')
        text = text.strip()
    return text


text = preprocess_text()
# 按每60个字符分句
max_seq_length = 60
sentences = [text[i:i + max_seq_length] for i in range(0, len(text), max_seq_length)]

# # 创建训练数据集
input_texts = sentences[:-1]
target_texts = sentences[1:]

# 使用Tokenizer来构建词汇表
tokenizer = Tokenizer(char_level=True)  # 以字符级别来处理

tokenizer.fit_on_texts(input_texts + target_texts)

# 将文本转换为序列
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)

# 填充序列
input_sequences = tensorflow.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_seq_length,
                                                                        padding='post')
target_sequences = tensorflow.keras.preprocessing.sequence.pad_sequences(target_sequences, maxlen=max_seq_length,
                                                                         padding='post')

# 将目标序列转换为3D张量，以便Keras的LSTM处理
target_sequences = np.expand_dims(target_sequences, -1)

# 打印一些信息以验证
print(f"词汇表大小: {len(tokenizer.word_index)}")
print(f"输入序列的最大长度: {max_seq_length}")
print(f"目标序列的最大长度: {max_seq_length}")

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding

# 参数设置
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 256
latent_dim = 512

# 编码器
encoder_inputs = Input(shape=(max_seq_length,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# 解码器
decoder_inputs = Input(shape=(max_seq_length,))
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)
decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 定义模型
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 编译模型
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# 打印模型结构
model.summary()

# 编码器模型
encoder_model = Model(encoder_inputs, encoder_states)

# 解码器的状态输入
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# 解码器输入（形状为1的单字符输入）
decoder_single_input = Input(shape=(1,))
dec_emb2 = dec_emb_layer(decoder_single_input)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

# 解码器模型
decoder_model = Model(
    [decoder_single_input] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)


# 文本生成函数
def decode_sequence(input_seq):
    # 编码输入序列为状态向量
    states_value = encoder_model.predict(input_seq)

    # 生成一个空的目标序列，长度为1
    target_seq = np.zeros((1, 1))

    # 以起始字符的索引作为目标序列的起始字符
    target_seq[0, 0] = tokenizer.word_index['我']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # 取概率最高的词作为下一个词
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = tokenizer.index_word.get(sampled_token_index, '')

        decoded_sentence += sampled_char

        # 如果达到停止条件或最大长度则停止
        if (sampled_char == '\n' or len(decoded_sentence) > max_seq_length):
            stop_condition = True

        # 更新目标序列
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # 更新状态
        states_value = [h, c]

    return decoded_sentence


# 生成样本文本函数
def generate_text(input_text):
    # 将输入文本转换为序列
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = tensorflow.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=max_seq_length, padding='post',
                                                                      truncating='post')

    # 调用解码函数生成后续文本
    generated_text = decode_sequence(input_seq)
    return generated_text


input_text = "青衣剑士连劈三剑，锦衫剑士一一格开。青衣剑士一声吒喝，长剑从左上角直划而下，势劲力急。锦衫剑士身手矫捷，向后跃开，避过了这剑。他左足刚着地，身子跟着弹起，刷刷两剑，向对手攻去。青衣剑士凝里不动，嘴角边微微冷笑，长剑轻摆，挡开来剑。"

epochs = 100
for i in range(epochs):
    # 训练模型
    model.fit(
        [input_sequences, input_sequences],
        target_sequences,
        batch_size=64,
        epochs=20
    )

    # 示例输入文本
    generated_text = generate_text(input_text)
    print(i)
    print(generated_text)
