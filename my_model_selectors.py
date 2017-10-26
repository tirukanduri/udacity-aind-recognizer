import math
import statistics
import warnings
import traceback
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    n: num_data_points
    p: num_free_params
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        https://en.wikipedia.org/wiki/Bayesian_information_criterion
        """
        # print("testing")
        warnings.filterwarnings("ignore", category=DeprecationWarning)


        best_num_components = 0
        best_BIC = float("+inf")

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # equation is 2 parts.
                # part 1:
                # https://discussions.udacity.com/t/bayesian-information-criteria-equation/326887/4
                #model.n_features is not coming up
                # There was supposed to be a variable n_features. But this is giving error \
                # So not using it.
                #print("n_features: ",model.n_features )

                # This corresponds to p in the formula
                num_free_params = (n ** 2) + 2 * n * len(self.X[0]) -1

                # part 2
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                bic = (-2 * logL) + num_free_params * (np.log(len(self.X[:,0])))
                if bic < best_BIC:
                    best_BIC = bic
                    best_num_components = n
            except Exception as e:

                pass


        if best_num_components == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)

    # raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # raise NotImplementedError

        best_num_components = 0
        best_DIC = float("-inf")
        # print("HWords: ", self.hwords)
        other_words = [self.hwords[word] for word in self.words if word != self.this_word]

        for n in range(self.min_n_components, self.max_n_components + 1):

            try:
                log = self.base_model(n).score(self.X, self.lengths)
                sum = 0.
                for word in other_words:
                    sum += self.base_model(n).score(word[0], word[1])

                DIC = log - sum / (len(other_words))
                if (DIC > best_DIC):
                    best_DIC = DIC
                    best_num_components = n
            except:
                pass

        if best_num_components == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # raise NotImplementedError

        best_num_components = 0
        best_AVG = float("-inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                if len(self.sequences)>2:
                    kfold = KFold(n_splits=3)
                    logs = []
                    for train, test in kfold.split(self.sequences):
                        try:
                            trainX,train_lengths = combine_sequences(train, self.sequences)
                            testX,test_lengths = combine_sequences(test,self.sequences)
                            model = GaussianHMM(n_components=n, covariance_type="diag",\
                                                n_iter=1000, random_state=self.random_state,\
                                                verbose=False).fit(trainX,train_lengths)
                            #model= self.base_model(len(trainX)).fit(trainX,train_lengths)
                            score = model.score(testX,test_lengths)
                            logs.append(score)
                        except:
                            pass

                    log_avg = sum(logs) / max(1,len(logs))
                    if log_avg > best_AVG:
                        best_AVG = log_avg
                        best_num_components = n
                else:
                    score = self.base_model(n). score(self.X, self.lengths)
                    if score > best_AVG:
                        best_AVG = score
                        best_num_components = n

            except Exception as e:
                pass

        if best_num_components == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_num_components)
