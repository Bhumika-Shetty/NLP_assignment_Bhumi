import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    # Implement hybrid transformation: synonym replacement + typos
    text = example["text"]
    
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Initialize detokenizer for reconstructing the sentence
    detokenizer = TreebankWordDetokenizer()
    
    # QWERTY keyboard layout for typo simulation
    keyboard_neighbors = {
        'a': ['q', 'w', 's', 'z'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 's', 'd', 'r'], 'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'], 'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'j', 'k', 'o'],
        'j': ['h', 'u', 'i', 'k', 'n', 'm'], 'k': ['j', 'i', 'o', 'l', 'm'], 'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'], 'o': ['i', 'k', 'l', 'p'], 'p': ['o', 'l'],
        'q': ['w', 'a', 's'], 'r': ['e', 'd', 'f', 't'], 's': ['a', 'w', 'e', 'd', 'z', 'x'],
        't': ['r', 'f', 'g', 'y'], 'u': ['y', 'h', 'j', 'i'], 'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'a', 's', 'e'], 'x': ['z', 's', 'd', 'c'], 'y': ['t', 'g', 'h', 'u'],
        'z': ['a', 's', 'x']
    }
    
    # Process each token for potential transformations
    transformed_tokens = []
    
    for token in tokens:
        # Only process alphabetic words (skip punctuation, numbers)
        if token.isalpha() and len(token) > 2:  # Skip very short words for better readability
            
            # Step 1: Try synonym replacement (50% probability)
            if random.random() < 0.5:
                # Get synsets (synonym sets) for the word
                synsets = wordnet.synsets(token.lower())
                
                if synsets:
                    # Collect all possible synonyms from all synsets
                    synonyms = []
                    for synset in synsets:
                        for lemma in synset.lemmas():
                            synonym = lemma.name()
                            # Only add if it's different from original and doesn't contain underscores
                            if synonym.lower() != token.lower() and '_' not in synonym:
                                synonyms.append(synonym)
                    
                    # If we found synonyms, randomly choose one
                    if synonyms:
                        chosen_synonym = random.choice(synonyms)
                        # Preserve original capitalization pattern
                        if token.isupper():
                            chosen_synonym = chosen_synonym.upper()
                        elif token.istitle():
                            chosen_synonym = chosen_synonym.capitalize()
                        else:
                            chosen_synonym = chosen_synonym.lower()
                        
                        transformed_tokens.append(chosen_synonym)
                        continue  # Skip to next token, already transformed
            
            # Step 2: Try typo introduction (20% probability, only if not already transformed)
            if random.random() < 0.2 and len(token) > 3:  # Only apply typos to longer words
                token_chars = list(token)
                # Randomly select a position to introduce typo (avoid first and last character)
                if len(token_chars) > 3:
                    typo_pos = random.randint(1, len(token_chars) - 2)
                    original_char = token_chars[typo_pos].lower()
                    
                    # If character has keyboard neighbors, replace with one
                    if original_char in keyboard_neighbors:
                        new_char = random.choice(keyboard_neighbors[original_char])
                        # Preserve capitalization
                        if token_chars[typo_pos].isupper():
                            new_char = new_char.upper()
                        token_chars[typo_pos] = new_char
                        
                        transformed_tokens.append(''.join(token_chars))
                        continue  # Skip to next token, already transformed
            
            # No transformation applied, keep original
            transformed_tokens.append(token)
        else:
            # Non-alphabetic token or very short word, keep as is
            transformed_tokens.append(token)
    
    # Reconstruct the sentence from tokens
    transformed_text = detokenizer.detokenize(transformed_tokens)
    
    # Update the example with transformed text
    example["text"] = transformed_text

    ##### YOUR CODE ENDS HERE ######

    return example