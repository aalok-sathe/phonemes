__author__ = 'Anton Melnikov' + ' and Aalok Sathe and Shinjini Ghosh'


from collections import Counter, OrderedDict, defaultdict
from enum import Enum
from itertools import chain
from pprint import pprint

from parsimonious.grammar import Grammar
from parsimonious import IncompleteParseError


ipaexpr = Grammar(r'''
     SOUND = CONSONANT / VOWEL / OTHER
         CONSONANT = (STYLE SP)* PLACE (SP "or" SP PLACE)? (SP LATERAL)? SP MANNER (SP "or" SP MANNER)? (SP RELEASE)?
             STYLE = EJECTIVE / ASPIRATION / VOICE / TENDENCY
                 EJECTIVE = "ejective"
                 ASPIRATION = "aspirated"
                 VOICE = "voiced" / "voiceless"
                 TENDENCY = "labialized" / "palatalised"
             PLACE = "labial-palatal" / "alveolo-palatal" / "palato-alveolar" / "velar" / "labial" / "dental" / "alveolar" / "uvular" / "palatal" / "retroflex" / "labiodental" / "labiovelar" / "bilabial" / "glottal" / "pharyngeal" / "postalveolar" / "syllabic"
             LATERAL = "lateral"
             MANNER = "stop" / "fricative" / "approximant" / "approximate" / "implosive" / "plosive" / "click" / "nasal" / "trill" / "tap" / "flap" / "affricate" / "sibilant"
             RELEASE = "release"

     VOWEL = NEAR? OPENING SP NEAR? POSITION SP ROUNDING SP "vowel"
         NEAR = "near-"
         OPENING = "high-mid" / "low-mid" / "high" / "mid" / "low"
         POSITION = "front" / "central" / "back"
         ROUNDING = "rounded" / "unrounded"

     OTHER = (~".+" SP?)*

     SP = ~"\s*"
         ''')

def parse(text):
    print(ipaexpr.parse(text))


semiterm = {'EJECTIVE', 'ASPIRATION', 'VOICE', 'TENDENCY',
            'PLACE', 'LATERAL', 'MANNER',
            'RELEASE',
            'NEAR', 'OPENING', 'POSITION', 'ROUNDING',}


def flatten(tree, collection: defaultdict(list)):
    '''flattens a tree and picks out relevant properties
    (semi-terminals) specified in a set a priori

    example output in collection:
    '''
    # is it a semiterminal?
    if tree.expr_name in semiterm:
        expr_name = tree.expr_name
        while tree and not hasattr(tree.expr, 'literal'):
            tree = tree.children[0]
        collection[expr_name.lower()] += [tree.expr.literal]

    # mark broad category
    if tree.expr_name.lower() in {'vowel', 'consonant'}:
        collection[tree.expr_name.lower()] = [tree.expr_name.lower()]

    # make recursive calls till leafs encountered (empty children)
    for subtree in tree.children:
        flatten(subtree, collection)

    # create placeholder entry for all keys
    for key in semiterm.union({'vowel', 'consonant'}):
        collection[key.lower()]
    return collection


class AutoEnumCount:
    def __init__(self, n=1, inc=1):
        self.n = n
        self.inc = inc

    def mark(self, n=None, inc=1):
        if n:
            self.n = n
        self.n += self.inc
        return self.n - self.inc

ec = AutoEnumCount()

Sonority = Enum('Sonority', {' ': ec.mark(),

                             "consonant": ec.mark(10),
                             "voiceless": ec.mark(), "voiced": ec.mark(),
                             "click": ec.mark(100, 10), "implosive": ec.mark(),
                             "stop": ec.mark(), "plosive": ec.mark(),
                             "affricate": ec.mark(), "fricative": ec.mark(),
                             "sibilant": ec.mark(), "nasal": ec.mark(),
                             "trill": ec.mark(), "lateral": ec.mark(),
                             "tap": ec.mark(), "flap": ec.mark(),
                             "approximant": ec.mark(), "approximate": ec.mark(),

                             "vowel": ec.mark(1000, 100),
                             "low": ec.mark(), "low-mid": ec.mark(),
                             "mid": ec.mark(), "high-mid": ec.mark(),
                             "high": ec.mark()})


class FeatureValue(Enum):
    """
    enum for values of phonological features
    """
    yes = 1
    no = 0
    both = 2
    unspecified = -1


class FeatureValueDict(OrderedDict):
    pass


class Phoneme:

    def __init__(self, symbol, name, features, info, is_complete=True,
                 parent_phonemes: set = None, feature_counter: Counter = None,
                 parent_similarity=1.0, **kwargs):
        """
        :param is_complete: indicates whether the object represents a complete phoneme
        """
        self.value = self.parse_features(features)

        self.symbol = symbol
        self.name = name
        self.properties = info or {}
        # self.ipa_desc = self.parse_ipa(info['ipa description'])
        self.ipa_desc = self.parse_ipa(name)

        self.properties['classes'] = ' '.join(self.ipa_desc.values())
        self.properties['sonority'] = self.sonority()
        self.properties.update(kwargs)

        self.is_complete = is_complete

        if parent_phonemes:
            self.parent_phonemes = parent_phonemes
        if not parent_phonemes:
            self.parent_phonemes = {symbol}

        # the count of how similar the parent phonemes are
        self.parent_similarity = parent_similarity

    def __repr__(self):
        return self.symbol

    def __str__(self):
        return self.symbol

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        if other:
            return self.value == other.value
        else:
            # the other must be None
            return False

    def __len__(self):
        return len(self.value)

    def __contains__(self, item):
        return item in self.value

    def __iter__(self):
        return iter(self.value.items())

    @classmethod
    def from_symbol(cls, symbol: str, phonemes: dict):
        """
        Initialise a Phoneme object from its IPA symbol, using a dictionary of IPA symbols and features
        :param symbol:
        :param phonemes:
        :return:
        """
        phoneme = phonemes[symbol]
        name = phoneme['name']
        features = cls.parse_features(phoneme['features'])
        info = None#phoneme['info']
        # return cls(symbol, name, features, info, **phoneme)
        return cls(symbol, info=None, **phoneme)

    @staticmethod
    def parse_features(features_dict) -> FeatureValueDict:

        if isinstance(features_dict, FeatureValueDict):
            return features_dict

        features = FeatureValueDict()
        for feature, value in features_dict.items():

            # values can be True, False, 0 or ±

            if value is True:
                feature_value = FeatureValue.yes
            elif value is False:
                feature_value = FeatureValue.no
            elif value == 0:
                feature_value = FeatureValue.unspecified
            elif value == '±':
                feature_value = FeatureValue.both
            else:
                raise ValueError('{} is not recognised'.format(value))

            features[feature] = feature_value

        return features

    @staticmethod
    def parse_ipa(ipa_desc):
        '''
        parses an ipa description according to a gammar and returns a dictionary
        description of the phoneme in the form
            ...
            'voice': 'voiced',
            'place': 'bilabial',
            'manner': 'plosive'
        ...etc
        '''
        classes = defaultdict(str)
        features = ipa_desc.lower()
        tree = ipaexpr.parse(features)
        classes.update({k: ' '.join(v) for k, v in flatten(tree, defaultdict(list)).items()})

        return classes


    def sonority(self):
        '''computes a sonority score of this phoneme'''
        score = 0

        manner = self.ipa_desc['manner'].split() + [self.ipa_desc['lateral']]
        props = [self.ipa_desc['voice'] or 'voiceless', self.ipa_desc['consonant'],
                 self.ipa_desc['vowel'], self.ipa_desc['opening']]

        score += sum(Sonority[key or 'consonant'].value
                     for key in manner) / len(manner)
        score += sum([Sonority[key or ' '].value for key in props])

        return round(score**.4, 2)

    @property
    def features(self):
        return self.value

    def get_positive_features(self):
        for feature, value in self:
            if value == FeatureValue.yes or value == FeatureValue.both:
                yield feature

    def similarity_ratio(self, other):
        """
        computes the similarity between this Phoneme object and another
        :param other: Phoneme
        :return:
        """
        similarity_count = 0
        for feature, feature_value in self:
            other_feature = other.value[feature]

            if other_feature == feature_value:
                similarity_count += 1

            # add 0.5 if either of the features is ± and the other is + or -
            elif other_feature == FeatureValue.both or feature_value == FeatureValue.both:
                if (other_feature != FeatureValue.unspecified
                        and feature_value != FeatureValue.unspecified):
                    similarity_count += 0.5

        similarity_ratio = similarity_count / len(self.features)
        return similarity_ratio

    def partial_equals(self, other, threshold=0.7):
        """
        returns True if this Phoneme object's similarity to another Phoneme object
        is equal to or above the given threshold of similarity
        :param other: Phoneme
        :param threshold: similarity threshold
        :return:
        """
        similarity_ratio = self.similarity_ratio(other)

        if similarity_ratio >= threshold:
            return True
        else:
            return False

    def intersection(self, other):
        """
        Returns an 'intersection phoneme' between this Phone object and another
        :param other: Phoneme
        :return: Phoneme
        """
        if self == other:
            return self
        elif other:
            if other.symbol in self.parent_phonemes:
                return self

            intersection = FeatureValueDict(set(self).intersection(set(other)))

            # create new parents
            new_parents = set(chain(self.parent_phonemes, other.parent_phonemes))

            new_symbol = '/'.join(new_parents)

            combined_similarity = self.similarity_ratio(other)

            partial_phoneme = Phoneme(new_symbol, 'partial phoneme',
                                      intersection, is_complete=False,
                                      parent_phonemes=new_parents,
                                      parent_similarity=combined_similarity)
            return partial_phoneme

        else:
            return None

    def pick_closest(self, other_phonemes):
        """
        Picks the closest Phoneme object (using the similarity ratio) from an iterable of Phoneme objects
        :param other_phonemes: iterable of Phonemes
        :return: Phoneme
        """
        closest = max(other_phonemes, key=lambda phoneme: self.similarity_ratio(phoneme))
        return closest
