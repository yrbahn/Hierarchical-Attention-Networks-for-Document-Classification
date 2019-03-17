
class Vocab(object):
    def __init__(self, vocab_file):
        self.vocab = ['PAD', 'UNK']
        self.word2id = {'PAD': 0, 'UNK': 1}
        self.index = len(self.vocab)
        self._load_word2id(vocab_file)

    def _load_word2id(self, vocab_file, threshold=None):
        vf = open(vocab_file)
        for line in vf:
            word, freq = line.split()
            freq = int(freq)
            if threshold == None or freq >= threshold:
                if word not in self.vocab:
                    self.vocab.append(word)
                    self.word2id[word] = self.index
                    self.index = self.index + 1

    def get_size(self):
        return len(self.vocab)

    def get_ids(self, word_list):
        return list(map(lambda word: self.get_id(word), word_list))

    def get_id(self, word):
        if word in self.vocab:
            return self.word2id[word]
        else:
            return 1 # UNK

