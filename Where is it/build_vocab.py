class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<UNK>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)
    


def build_vocab():
    vocab = Vocabulary()
    vocab.add_word('<PAD>')
    vocab.add_word('<BOS>')    
    vocab.add_word('<EOS>')
    vocab.add_word('<UNK>')
    for data in ['K','k',"N",'n','p','P','Q','q','-','B','b','R','r','1','2','3','4','5','6','7','8']:
        for  label in data:
            for c in label:
                vocab.add_word(c)
    return vocab

