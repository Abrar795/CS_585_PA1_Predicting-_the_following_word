import nltk
from nltk.corpus import brown, stopwords
from nltk.util import bigrams
from collections import Counter

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the Brown corpus
nltk.download('brown')

# Load stopwords
stop_words = set(stopwords.words('english'))

# Function to remove stopwords and punctuation from text
def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    return [word for word in words if word.isalnum() and word not in stop_words]

# Function to calculate bigram probabilities
def calculate_bigram_probabilities(bigram_list):
    bigram_counts = Counter(bigram_list)
    total_bigrams = len(bigram_list)
    bigram_probabilities = {}

    # Calculate probabilities
    for bigram, count in bigram_counts.items():
        bigram_probabilities[bigram] = count / total_bigrams

    return bigram_probabilities

# Ask the user for the initial word/token
while True:
    initial_word = input("Enter the initial word/token: ").lower()
    if initial_word in stop_words:
        print("Please enter a non-stopword.")
        continue
    if initial_word not in brown.words():
        print("The word is not in the corpus.")
        choice = input("Would you like to try again? (y/n): ").lower()
        if choice != 'y':
            break
        continue
    break

# Extract bigrams from the Brown corpus
brown_corpus = brown.words()
brown_bigrams = list(bigrams([token.lower() for token in brown_corpus]))

# Filter bigrams based on the initial word
filtered_bigrams = [bigram for bigram in brown_bigrams if bigram[0] == initial_word]

# Calculate probabilities for bigrams starting with the initial word
initial_word_bigram_probabilities = calculate_bigram_probabilities(filtered_bigrams)

# Sort bigrams based on probabilities
sorted_bigrams = sorted(initial_word_bigram_probabilities.items(), key=lambda x: x[1], reverse=True)

# Initialize the sentence
sentence = [initial_word]

# Display top 3 most likely words to follow the initial word
while True:
    print(f"\n{initial_word} ...")
    print("Which word should follow:")
    for i, (bigram, probability) in enumerate(sorted_bigrams[:3], start=1):
        print(f"{i}) {bigram[1]} P({initial_word} {bigram[1]}) = {probability:.2f}")
    print("4) QUIT")

    user_choice = input("Enter your choice: ")

    if user_choice == '4':
        break
    elif user_choice not in ['1', '2', '3']:
        user_choice = '1'

    next_word = sorted_bigrams[int(user_choice) - 1][0][1]
    sentence.append(next_word)
    initial_word = next_word
    filtered_bigrams = [bigram for bigram in brown_bigrams if bigram[0] == initial_word]
    initial_word_bigram_probabilities = calculate_bigram_probabilities(filtered_bigrams)
    sorted_bigrams = sorted(initial_word_bigram_probabilities.items(), key=lambda x: x[1], reverse=True)

# Join the selected words to form the full sentence
full_sentence = ' '.join(sentence)
print("Full Sentence:", full_sentence)
print("Quitting...")
