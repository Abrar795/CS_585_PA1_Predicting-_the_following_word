import nltk
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter

# Downloading NLTK resources
nltk.download('brown')

# Extracting two-grams from the Brown corpus
brown_words = brown.words()
bigrams = list(ngrams(brown_words, 2))

# Calculating frequencies of each bigram
bigram_frequency = Counter(bigrams)

# Calculate total number of bigrams
total_bigrams_count = sum(bigram_frequency.values())

# Calculate the vocabulary size
vocab_size = len(set(brown_words))

# Calculate probabilities of each bigram with Laplace smoothing (add-one smoothing)
bigram_prob = {bigram: (bigram_frequency[bigram] + 1) / (total_bigrams_count + vocab_size) for bigram in bigram_frequency}

# Ask the user to enter a sentence
sentence = input("Please enter a sentence: ")

# Apply lowercasing to the input sentence
sentence_lower = sentence.lower()

# Tokenize the sentence into words
words = sentence_lower.split()

# Generate bigrams from the tokenized sentence
sentence_bigrams = list(ngrams(words, 2))

# Calculate the probability of the sentence using 2-gram model
sentence_prob = 1.0
for bigram in sentence_bigrams:
    if bigram in bigram_prob:
        sentence_prob *= bigram_prob[bigram]
    else:
        # Apply Laplace smoothing for unseen bigrams
        sentence_prob *= 1 / (total_bigrams_count + vocab_size)

# Display the original sentence
print("\nOriginal Sentence:", sentence)

# Display individual bigrams and their probabilities
print("\nIndividual Bigrams and Their Probabilities:")
for bigram in sentence_bigrams:
    if bigram in bigram_prob:
        print(f"{bigram}: {bigram_prob[bigram]}")
    else:
        # Display Laplace smoothed probability for unseen bigrams
        print(f"{bigram}: {1 / (total_bigrams_count + vocab_size)}")

# Display the final probability P(S)
print("\nFinal Probability P(S):", sentence_prob)
