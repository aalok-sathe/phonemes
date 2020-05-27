__author__ = 'Anton Melnikov' + ' and Aalok Sathe and Shinjini Ghosh'

from collections import OrderedDict, defaultdict
import json
from pathlib import Path

from .phoneme import Phoneme, FeatureValue

this_path = Path(__file__)
phonemes_path = Path(this_path.parent.parent, 'phonemes.json')
vowels_path = Path(this_path.parent.parent, 'vowelspace.json')
ipabook_path = Path(this_path.parent.parent, 'ipabook.json')

# phoneme_dict = defaultdict(dict)
with phonemes_path.open() as fp:
    phoneme_dict = (json.load(fp, object_pairs_hook=OrderedDict))


with vowels_path.open() as fp:
    for k,v in json.load(fp, object_pairs_hook=OrderedDict).items():
        if k in phoneme_dict: phoneme_dict[k].update(v)

# with ipabook_path.open() as fp:
#     ipa_dict = json.load(fp, object_pairs_hook=OrderedDict)
# ipa_dict = {entry['Character']: {k.lower(): v for k, v in entry.items()}
#             for entry in ipa_dict}

# merged_dict = {sym: dict(**phoneme_dict[sym], info=ipa_dict[sym])
#                for sym in phoneme_dict.keys() & ipa_dict.keys()}

phonemes = {symbol: Phoneme.from_symbol(symbol, phoneme_dict)
            for symbol in phoneme_dict}
