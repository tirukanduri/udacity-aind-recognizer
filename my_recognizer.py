import warnings
from asl_data import SinglesData
from hmmlearn.hmm import GaussianHMM

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    #raise NotImplementedError
    for test_item in test_set.get_all_sequences():
        X,lengths = test_set.get_item_Xlengths(test_item)

        best_prob=float("-inf")
        best_word=None
        score={}
        for word,model in models.items():
            try:
                prob =  model.score(X,lengths)
                score[word]=prob
                if(prob>best_prob):
                    best_prob=prob
                    best_word=word
            except:
                score[word]=float("-inf")
                pass

        probabilities.append(score)
        guesses.append(best_word)

    return probabilities,guesses