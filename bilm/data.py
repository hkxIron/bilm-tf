# originally based on https://github.com/tensorflow/models/tree/master/lm_1b
import glob
import random

import numpy as np

from typing import List

"""
数据准备包括:
1.生成word的词汇表类; 
2.生成字符的词汇表类； 
3.以word-ids作为输入的训练batch生成类; 
4.以char-ids作为输入的训练batch生成类; 
5.生成语言模型输入的数据集类
"""

# 1.1 word词汇表类(Vocabulary)
class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.

    '''
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).

            vocab文件，是一个纯文本，每一行只有一个词。另外，这个文件应该包含特殊词，
            比如<S>, </S>, <UNK>等
        '''
        self._id_to_word = [] # word的list,由于加载的是vocab,因此word 不会重复
        self._word_to_id = {}
        self._unk_word_id = -1
        self._bos_word_id = -1
        self._eos_word_id = -1

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos_word_id = idx # 0 -> <s>
                elif word_name == '</S>':
                    self._eos_word_id = idx # 1 -> </s>
                elif word_name == '<UNK>':
                    self._unk_word_id = idx # 2 -> <UNK>
                if word_name == '!!!MAXTERMID': # 超过最大行数
                    continue

                self._id_to_word.append(word_name) # 数据的下标就是id
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos_word_id == -1 or self._eos_word_id == -1 or self._unk_word_id == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos_word_id(self):
        return self._bos_word_id

    @property
    def eos_word_id(self):
        return self._eos_word_id

    @property
    def unk_word_id(self):
        return self._unk_word_id

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk_word_id

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        #  将一个ids序列转化为word序列
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode_sentence_to_token_ids(self, sentence, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.

        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately.
            将一个sentenct转化为ids序列,并提供句子反转的功能

            :param sentence:"i love nlp"
            :return [123, 45, 6]
            """

        if split:
            word_ids = [ self.word_to_id(cur_word) for cur_word in sentence.split() ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos_word_id] + word_ids + [self.bos_word_id], dtype=np.int32) # 不应该里面的语序都是反的么
        else:
            return np.array([self.bos_word_id] + word_ids + [self.eos_word_id], dtype=np.int32)

# 1.2 字符词汇表(UnicodeCharsVocabulary)
"""
注意这个类是上面word词汇表Vocabulary的子类，这意味着这个字符类包含了Vocabulary的所有变量和方法！ 

每个字符(character)的id是用该字符对应的utf-8编码，
这样也就可以形成id和char之间的转换，因为使用utf-8编码，
这将限制char词汇表中所有可能的id数量为256。当然，
我们也需要加入5个额外的特殊字符，
包括:句首，句尾，词头，词尾和padding.
"""
class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.

    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.

    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.

    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_char_count_in_token, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_char_count_in_token = max_char_count_in_token

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars

        # sentence
        self.bos_char_id = 256  # <begin sentence>
        self.eos_char_id = 257  # <end sentence>
        # word
        self.bow_char_id = 258  # <begin word>
        self.eow_char_id = 259  # <end word>
        # padding
        self.pad_char_id = 260 # <padding>

        num_words = len(self._id_to_word) # vocab里最终有多少word

        self._word_id_to_char_ids = np.zeros([num_words, max_char_count_in_token],  # [vocab_size, max_word_length]
                                             dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos_char_ids(char_id):
            r = np.zeros([self.max_char_count_in_token], dtype=np.int32)
            r[:] = self.pad_char_id
            r[0] = self.bow_char_id # begin
            r[1] = char_id # index=1为char
            r[2] = self.eow_char_id # end
            # r[3:] = padding
            return r
        self.bos_char_ids = _make_bos_eos_char_ids(self.bos_char_id)
        self.eos_char_ids = _make_bos_eos_char_ids(self.eos_char_id)

        for i, word in enumerate(self._id_to_word):
            #word -> char_ids
            self._word_id_to_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_id_to_char_ids[self.bos_word_id] = self.bos_char_ids
        self._word_id_to_char_ids[self.eos_word_id] = self.eos_char_ids
        # TODO: properly handle <UNK>

    @property
    def word_id_to_char_ids_dict(self):
        return self._word_id_to_char_ids

    @property
    def max_char_count_in_token(self):
        return self._max_char_count_in_token

    # 将词转化为char_ids
    # "丕".encode("utf-8"), utf8编码为: E4B895, 即 b'\xe4\xb8\x95', 将会拆为3个char_id,即 228, 184, 149
    # "两".encode("utf-8"), utf8编码为: E4B8A4, 即 b'\xe4\xb8\xa4', 将会拆为3个char_id,即 228, 184, 164
    # "严".encode("utf-8"), utf8编码为: E4B8A5, 即 b'\xe4\xb8\xa5', 将会拆为3个char_id,即 228, 184, 165
    # 但对于中文而言, "两"与"严"虽然前面两个字符相同,却并没有任何相似性
    # 因此,此处的 word,对中文而言是 "严格" 而非单字 "严"
    # >>> "严格".encode("utf-8")
    # b'\xe4\xb8\xa5\xe6\xa0\xbc', 但感觉这样,也是有上面的问题!!!

    # "token".encode("utf-8"), utf8编码为:token, 即 b'token', 将会拆为5个char_id,即 't','o','k','e','n'
    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_char_count_in_token], dtype=np.int32)
        code[:] = self.pad_char_id
        # "严".encode("utf-8"), utf8编码为: E4B8A5, 即 b'\xe4\xb8\xa5'
        # "token".encode("utf-8"), utf8编码为:token, 即 b'token'
        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_char_count_in_token - 2)] # 这个max_word_length中已经包含start与end
        code[0] = self.bow_char_id
        for k, chr_id in enumerate(word_encoded, start=1): # 从1开始计数
            code[k] = chr_id
        code[len(word_encoded) + 1] = self.eow_char_id

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_id_to_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_sentence_to_char_ids(self, sentence, reverse=False, need_split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if need_split:
            chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word) for cur_word in sentence]
        # [token_count, max_char_count_in_token]
        if reverse:
            return np.vstack([self.eos_char_ids] + chars_ids + [self.bos_char_ids])
        else:
            return np.vstack([self.bos_char_ids] + chars_ids + [self.eos_char_ids])


class CharBatcher(object): # change to char_id matrix
    ''' 
    Batch sentences of tokenized text into character id matrices.
    '''
    def __init__(self, lm_vocab_file: str, max_char_count_in_token: int):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        max_token_length = the maximum number of characters in each token
        '''
        self._lm_vocab = UnicodeCharsVocabulary(
            lm_vocab_file, max_char_count_in_token
        )
        self._max_char_count_in_token = max_char_count_in_token

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_token_count_in_sentence = max(len(sentence) for sentence in sentences) + 2 # sos, eos word
        # X_char_ids: [batch, max_token_count_in_sentence, max_char_count_in_token]
        X_char_ids = np.zeros(
            (n_sentences, max_token_count_in_sentence, self._max_char_count_in_token),
            dtype=np.int64
        )

        for k, sent in enumerate(sentences):
            length = len(sent) + 2 # sos, eos word
            # ex:"we are reading" -> ['w', 'e', ' ', 'a', 'r','e' ...]
            # char_ids_without_mask:[token_count, max_char_count_in_token]
            char_ids_without_mask = self._lm_vocab.encode_sentence_to_char_ids(
                sent, need_split=False)

            # X_char_ids: [batch, max_token_count_in_sentence, max_char_count_in_token]
            # add one so that 0 is the mask value
            # 注意:此处在每个id上加了1,还原的时候,需要减回来, 感觉此种做法并不好,应该直接在vocab中将mask列为第一个
            X_char_ids[k, :length, :] = char_ids_without_mask + 1

        return X_char_ids


class TokenBatcher(object): # change to token id matrix
    ''' 
    Batch sentences of tokenized text into token id matrices.
    '''
    def __init__(self, lm_vocab_file: str):
        '''
        lm_vocab_file = the language model vocabulary file (one line per
            token)
        '''
        self._lm_vocab = Vocabulary(lm_vocab_file)

    def batch_sentences(self, sentences: List[List[str]]):
        '''
        Batch the sentences as character ids
        Each sentence is a list of tokens without <s> or </s>, e.g.
        [['The', 'first', 'sentence', '.'], ['Second', '.']]
        '''
        n_sentences = len(sentences)
        max_token_count_in_sentence = max(len(sentence) for sentence in sentences) + 2
        # [batch, max_token_count_in_sentence]
        X_ids = np.zeros((n_sentences, max_token_count_in_sentence), dtype=np.int64)

        for k, sent in enumerate(sentences):
            length = len(sent) + 2
            ids_without_mask = self._lm_vocab.encode_sentence_to_token_ids(sent, split=False)
            # add one so that 0 is the mask value
            X_ids[k, :length] = ids_without_mask + 1

        # [batch, max_token_count_in_sentence]
        return X_ids


##### for training
# generator: 从文件中生成 (token_ids, char_ids),每条样本一行记录,如 函数get_sentences()
# 返回: token_inputs, char_inputs, target_tokens
def _get_batch(generator, batch_size, num_steps, max_char_count_in_word):
    """Read batches of input."""
    cur_stream = [None] * batch_size

    no_more_data = False
    while True:
        # token_inputs:[batch_size, num_steps]
        token_inputs = np.zeros([batch_size, num_steps], np.int32)
        if max_char_count_in_word:
            char_inputs = np.zeros([batch_size, num_steps, max_char_count_in_word],
                                   np.int32)
        else:
            char_inputs = None
        target_tokens = np.zeros([batch_size, num_steps], np.int32)

        for i in range(batch_size):
            cur_pos = 0

            while cur_pos < num_steps: # 时间步
                # cur_steam[i]中数据为空,需要重新从generater中获取
                if cur_stream[i] is None or len(cur_stream[i][0]) <= 1:
                    try:
                        # cur_steam[i]:[
                        #   (token_ids:[sentence_count, token_count_per_sentence],
                        #    char_ids:[sentence_count, token_count_per_sentence, char_count_per_token])
                        #   ...
                        #   (token_ids:[sentence_count, token_count_per_sentence],
                        #    char_ids:[sentence_count, token_count_per_sentence, char_count_per_token])
                        # ]
                        cur_stream[i] = list(next(generator)) # 获取下一个句子作为记录
                    except StopIteration:
                        # No more data, exhaust current streams and quit
                        no_more_data = True
                        break

                # cur_stream[i][0]: token_ids, cur_stream[i][1]:char_ids
                token_count = len(cur_stream[i][0]) - 1
                token_offset = min(token_count, num_steps - cur_pos)
                next_pos = cur_pos + token_offset

                token_inputs[i, cur_pos:next_pos] = cur_stream[i][0][:token_offset]
                if max_char_count_in_word:
                    char_inputs[i, cur_pos:next_pos] = cur_stream[i][1][:token_offset]

                # target需要偏移一位
                target_tokens[i, cur_pos:next_pos] = cur_stream[i][0][1:token_offset + 1]
                cur_pos = next_pos
                # 将句子中剩余的句子作为下一个time_step的数据
                cur_stream[i][0] = cur_stream[i][0][token_offset:]
                if max_char_count_in_word:
                    cur_stream[i][1] = cur_stream[i][1][token_offset:]

        if no_more_data:
            # There is no more data.  Note: this will not return data
            # for the incomplete batch
            break

        # token_ids:[sentence_count, token_count_per_sentence],
        # tokens_characters:[sentence_count, token_count_per_sentence, char_count_per_token],
        # next_token_id:[sentence_count, token_count_per_sentence]
        X = {'token_ids': token_inputs,
             'tokens_characters': char_inputs,
             'next_token_id': target_tokens}

        yield X

class LMDataset(object):
    """
    Hold a language model dataset.

    A dataset is a list of tokenized files.  Each file contains one sentence
        per line.  Each sentence is pre-tokenized and white space joined.
    """
    def __init__(self, filepattern, vocab, reverse=False, test=False,
                 shuffle_on_load=False):
        '''
        filepattern = a glob string that specifies the list of files.
        vocab = an instance of Vocabulary or UnicodeCharsVocabulary
        reverse = if True, then iterate over tokens in each sentence in reverse
        test = if True, then iterate through all data once then stop.
            Otherwise, iterate forever.
        shuffle_on_load = if True, then shuffle the sentences after loading.
        '''
        self._vocab = vocab
        self._all_shards = glob.glob(filepattern) # 用正则表达式来加载所需文件 list
        print('Found %d shards at %s' % (len(self._all_shards), filepattern))
        self._shards_to_choose = []

        self._reverse = reverse
        self._test = test
        self._shuffle_on_load = shuffle_on_load
        self._use_char_inputs = hasattr(vocab, 'encode_sentence_to_char_ids')

        self._ids = self._load_random_shard_file()

    def _choose_random_shard(self):
        if len(self._shards_to_choose) == 0:
            self._shards_to_choose = list(self._all_shards)
            random.shuffle(self._shards_to_choose)
        shard_name = self._shards_to_choose.pop()
        return shard_name

    def _load_random_shard_file(self):
        """Randomly select a file and read it."""
        if self._test:
            if len(self._all_shards) == 0:
                # we've loaded all the data
                # this will propogate up to the generator in get_batch
                # and stop iterating
                raise StopIteration
            else:
                shard_name = self._all_shards.pop() # list, 按顺序取一个样本文件,每次pop,list会少一个元素
        else:
            # just pick a random shard
            shard_name = self._choose_random_shard()

        # token_ids:[sentence_count, token_count_per_sentence], 二维句子id列表
        # chars_ids:[sentence_count, token_count_per_sentence, char_count_per_token]
        token_ids_and_char_ids = self._load_shard(shard_name)
        self._i = 0
        self._nids = len(token_ids_and_char_ids) # nids:此文件中有多少样本
        return token_ids_and_char_ids

    def _load_shard(self, shard_name):
        """Read one file and convert to ids.

        Args:
            shard_name: file path.

        Returns:
            list of (token_id, char_id) tuples.
            其中:
            token_ids:[sentence_count, token_count_per_sentence], 二维句子id列表
            chars_ids:[sentence_count, token_count_per_sentence, char_count_per_token]
        """
        print('Loading data from: %s' % shard_name)
        with open(shard_name) as f:
            sentences_raw = f.readlines()

        if self._reverse:
            sentences = []
            for sentence in sentences_raw:
                splitted = sentence.split()
                # 如果为逆序
                splitted.reverse()
                sentences.append(' '.join(splitted))
        else:
            sentences = sentences_raw

        if self._shuffle_on_load:
            random.shuffle(sentences)
        # token_ids:[sentence_count, token_count_per_sentence], 二维句子id列表
        token_ids = [self.vocab.encode_sentence_to_token_ids(sentence, self._reverse)
               for sentence in sentences]
        if self._use_char_inputs:
            # chars_ids:[sentence_count, token_count_per_sentence, char_count_per_token]
            chars_ids = [self.vocab.encode_sentence_to_char_ids(sentence, self._reverse)
                     for sentence in sentences]
        else:
            chars_ids = [None] * len(token_ids)

        print('Loaded %d sentences.' % len(token_ids))
        print('Finished loading')
        # token_ids:[sentence_count, token_count_per_sentence], 二维句子id列表
        # chars_ids:[sentence_count, token_count_per_sentence, char_count_per_token]
        return list(zip(token_ids, chars_ids))

    def get_sentence(self):
        while True:
            if self._i == self._nids: # nids:该文件中样本数
                self._ids = self._load_random_shard_file()
            # token_ids:[sentence_count, token_count_per_sentence], 二维句子id列表
            # chars_ids:[sentence_count, token_count_per_sentence, char_count_per_token]
            token_ids_and_char_ids = self._ids[self._i]
            self._i += 1
            yield token_ids_and_char_ids

    @property
    def max_char_count_in_token(self):
        if self._use_char_inputs:
            return self._vocab.max_char_count_in_token
        else:
            return None

    def iter_batches(self, batch_size, num_steps):
        for X in _get_batch(generator=self.get_sentence(), # 获取 token_ids以及char_ids
                            batch_size=batch_size,
                            num_steps=num_steps,
                            max_char_count_in_word=self.max_char_count_in_token):

            # token_ids:[batch_size, token_count_per_sentence],
            # tokens_characters:[batch_size, token_count_per_sentence, char_count_per_token],
            # next_token_id:[batch_size, token_count_per_sentence]

            # token_ids = (batch_size, num_steps)
            # char_inputs = (batch_size, num_steps, max_char_count_per_token=50) of character ids
            # targets = word ID of next word (batch_size, num_steps)
            yield X

    @property
    def vocab(self):
        return self._vocab

class BidirectionalLMDataset(object):
    def __init__(self, filepattern, vocab, test=False, shuffle_on_load=False):
        '''
        bidirectional version of LMDataset
        '''
        self._data_forward = LMDataset(
            filepattern, vocab, reverse=False, test=test,
            shuffle_on_load=shuffle_on_load)
        self._data_reverse = LMDataset(
            filepattern, vocab, reverse=True, test=test,
            shuffle_on_load=shuffle_on_load)

    def iter_batches(self, batch_size, num_steps):
        max_char_count_in_word = self._data_forward.max_char_count_in_token
        """
          X = {'token_ids': token_inputs,
             'tokens_characters': char_inputs,
             'next_token_id': target_tokens}
        """
        for X, X_reverse in zip(
            _get_batch(self._data_forward.get_sentence(), batch_size,
                       num_steps, max_char_count_in_word),
            _get_batch(self._data_reverse.get_sentence(), batch_size,
                       num_steps, max_char_count_in_word)
            ):

            # 将X_reverse的值加入X
            for k, v in X_reverse.items():
                X[k + '_reverse'] = v

            yield X


class InvalidNumberOfCharacters(Exception):
    pass

