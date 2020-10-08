import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    
    # iterate over each text file
    for file in os.listdir(directory):
        # is text file
        if file[-3:] == 'txt':
            f = open(os.path.join(directory, file))
            files[file] = f.read()
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)
    n = len(tokens)

    words = []
    punctuation = string.punctuation
    stopwords = set(nltk.corpus.stopwords.words("english"))

    for i in range(n):
        token = tokens[i].lower()

        # skip token which are punctuation or stopwords
        if token in punctuation or token in stopwords:
            continue
        words.append(token)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    wordIDF = dict()

    totalDoc = len(documents)

    # words in all doc
    words = set()
    for doc in documents:
        wordSet = set(documents[doc])
        for word in wordSet:
            if word in wordIDF:
                wordIDF[word] += 1
            else:
                wordIDF[word] = 1

    # calculate idf of each word
    for word in wordIDF:
        wordIDF[word] = math.log(float(totalDoc) / float(wordIDF[word]))
    return wordIDF


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    filesList = list(files.keys())

    # sort files as per their tf-idf score
    def sortByTFIDF(file):
        score = 0
        for word in query:
            tf  = files[file].count(word)
            idf = idfs[word]
            score += tf * idf
        return score

    filesList.sort(key=sortByTFIDF, reverse=True)

    # return top n files
    return filesList[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    senList = list(sentences.keys())

    # sort sentences as per their tf-idf score
    def sortByIDF(sentence):
        score = 0
        density = 0
        n = len(sentences[sentence])
        for word in query:
            if word in sentences[sentence]:
                density += sentences[sentence].count(word)
                idf = idfs[word]
                score += idf
        density /= float(n)
        return (-score, -density)

    senList.sort(key=sortByIDF)
    return senList[:n]


if __name__ == "__main__":
    main()
