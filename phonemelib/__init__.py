__author__ = 'Anton Melnikov' + ' and Aalok Sathe and Shinjini Ghosh'

from collections import OrderedDict
import json
from pathlib import Path

from .phoneme import Phoneme, FeatureValue

this_path = Path(__file__)
phonemes_path = Path(this_path.parent.parent, 'phonemes.json')
ipabook_path = Path(this_path.parent.parent, 'ipabook.json')

with phonemes_path.open() as phonemes_file:
    phoneme_dict = json.load(phonemes_file, object_pairs_hook=OrderedDict)
    # todo make this a defaultdict (phoneme_dict) and decreases dependence
    # on this list

with ipabook_path.open() as ipabook_file:
    ipa_dict = json.load(ipabook_file, object_pairs_hook=OrderedDict)
ipa_dict = {entry['Character']: {k.lower(): v for k, v in entry.items()}
            for entry in ipa_dict}

merged_dict = {sym: dict(**phoneme_dict[sym], info=ipa_dict[sym])
               for sym in phoneme_dict.keys() & ipa_dict.keys()}

phonemes = {symbol: Phoneme.from_symbol(symbol, merged_dict)
            for symbol in merged_dict}
