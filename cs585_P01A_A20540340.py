import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.corpus import brown, reuters
from collections import Counter

# Download NLTK resources
nltk.download('brown')
nltk.download('reuters')
nltk.download('stopwords')

# Remove stopwords
stop_words = set(stopwords.words('english'))

def get_word_freq_distribution(corpus):
    words = [word.lower() for word in corpus.words() if word.lower() not in stop_words]
    freq_dist = Counter(words)
    return freq_dist

# Get word frequency distribution for Reuters corpus
reuters_freq_dist = get_word_freq_distribution(reuters)

# Get word frequency distribution for Brown corpus
brown_freq_dist = get_word_freq_distribution(brown)

# Calculating the top ten words for Brown corpus
top_words_brown = brown_freq_dist.most_common(10)

# Display top ten words for Brown corpus
print("Top 10 words in the Brown corpus:")
for i, (word, freq) in enumerate(top_words_brown, 1):
    print(f"Rank {i}: {word}")
    print(f"Frequency: {freq}")

# Calculating the top ten words for Reuters corpus
top_words_reuters = reuters_freq_dist.most_common(10)

# Display top ten words for Reuters corpus
print("\nTop 10 words in the Reuters corpus:")
for i, (word, freq) in enumerate(top_words_reuters, 1):
    print(f"Rank {i}: {word} ")
    print(f"Frequency: {freq}")

# Function to generate log(rank) vs log(frequency) plot
def plot_log_rank_vs_log_freq(freq_dist, corpus_name, color):
    ranks = range(1, len(freq_dist) + 1)
    frequencies = [freq for word, freq in freq_dist.most_common()]
    plt.figure()
    plt.loglog(ranks, frequencies, color=color)
    plt.title(f'Log(R) vs Log(F) Plot - {corpus_name}')
    plt.xlabel('Log(R)')
    plt.ylabel('Log(F)')
    plt.grid(True)
    plt.show()

# Generate log(rank) vs log(frequency) plot for Brown corpus
plot_log_rank_vs_log_freq(brown_freq_dist, 'Brown', 'red')

# Generate log(rank) vs log(frequency) plot for Reuters corpus
plot_log_rank_vs_log_freq(reuters_freq_dist, 'Reuters', 'orange')

# Function to calculate unigram occurrence probability for a given word
def calculate_unigram_prob(word, freq_dist):
    total_words = sum(freq_dist.values())
    word_freq = freq_dist[word]
    prob = word_freq / total_words
    return prob, word_freq, total_words

# Calculate unigram occurrence probability for "MachineLearning" in Brown corpus
machine_learning_prob_brown, machine_learning_count_brown, total_words_brown = calculate_unigram_prob('machinelearning', brown_freq_dist)

# Calculate unigram occurrence probability for "ArtificialIntelligence" in Brown corpus
ai_prob_brown, ai_count_brown, _ = calculate_unigram_prob('artificialintelligence', brown_freq_dist)

# Calculate unigram occurrence probability for "Sports" in Brown corpus
sports_prob_brown, sports_count_brown, _ = calculate_unigram_prob('sports', brown_freq_dist)

# Calculate unigram occurrence probability for "Cooking" in Brown corpus
cooking_prob_brown, cooking_count_brown, _ = calculate_unigram_prob('cooking', brown_freq_dist)

# Define a function to print formatted information
def print_probability_info(word, count, total_words, probability):
    print(f"Word: {word:<20} | Count: {count:<10} | Total Words: {total_words:<10} | Probability: {probability:<10}")

# Display probabilities in a better format for the Brown corpus
print("\n" + "-"*85)
print(f"{'Brown Corpus Unigram Occurrence Probabilities':^85}")
print("-"*85)
print_probability_info("MachineLearning", machine_learning_count_brown, total_words_brown, machine_learning_prob_brown)
print_probability_info("ArtificialIntelligence", ai_count_brown, total_words_brown, ai_prob_brown)
print_probability_info("Sports", sports_count_brown, total_words_brown, sports_prob_brown)
print_probability_info("Cooking", cooking_count_brown, total_words_brown, cooking_prob_brown)
print("-"*85)

# Calculate unigram occurrence probability for "MachineLearning" in Reuters corpus
machine_learning_prob_reuters, machine_learning_count_reuters, total_words_reuters = calculate_unigram_prob('machinelearning', reuters_freq_dist)

# Calculate unigram occurrence probability for "ArtificialIntelligence" in Reuters corpus
ai_prob_reuters, ai_count_reuters, _ = calculate_unigram_prob('artificialintelligence', reuters_freq_dist)

# Calculate unigram occurrence probability for "Sports" in Reuters corpus
sports_prob_reuters, sports_count_reuters, _ = calculate_unigram_prob('sports', reuters_freq_dist)

# Calculate unigram occurrence probability for "Cooking" in Reuters corpus
cooking_prob_reuters, cooking_count_reuters, _ = calculate_unigram_prob('cooking', reuters_freq_dist)

# Display probabilities in a better format for the Reuters corpus
print("\n" + "-"*85)
print(f"{'Reuters Corpus Unigram Occurrence Probabilities':^85}")
print("-"*85)
print_probability_info("MachineLearning", machine_learning_count_reuters, total_words_reuters, machine_learning_prob_reuters)
print_probability_info("ArtificialIntelligence", ai_count_reuters, total_words_reuters, ai_prob_reuters)
print_probability_info("Sports", sports_count_reuters, total_words_reuters, sports_prob_reuters)
print_probability_info("Cooking", cooking_count_reuters, total_words_reuters, cooking_prob_reuters)
print("-"*85)
