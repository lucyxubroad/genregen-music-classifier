import math 
import time
from stemming.porter2 import stem 
"""
https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/Lyrics/lyrics_to_bow.py



"""

# "testinglyrics.txt"
FILENAME_LYRICS = "lyrics_train.txt" 
FILENAME_GENRE = "genres.txt"
FILENAME_TEST = "lyrics_test.txt"
# "lyrics_test.txt"    #"testing_tester.txt"
accepted_genres = ["Pop_Rock", "Rap", "Blues", "Religious", "Country", "Jazz"]
FILENAMES = ["pop_rock.txt", "rap.txt", "blues.txt", "religious.txt", "country.txt", "jazz.txt"]
FILENAMES2 = ["pop_rock2.txt", "rap2.txt", "blues2.txt", "religious2.txt", "country2.txt", "jazz2.txt"]

# words that appeared in the top 100 words of every genre
COMMON_WORDS1 = ['how', 'here', 'yeah', 'back', 'at', 'he', 'ca', 'would', 'make', 'let', 'way', 
'say', 'take', 'want', 'got', 'from', 'if', 'was', 'never', 'see', 'they', 'get', 'down', 
'one', 'out', 'come', 'there', 'go', 'can', 'up', 'time', 'now', 'when', 'like', 'just', 
'with', 'what', 'love', 'but', 'no', 'know', 'this', 'so', 'have', 'be', 'for', 'will', 
'all', 'on', 'am', 'we', 'are', 'that', 'do', 'your', 'my', 'of', 'is', 'in', 'me', 
'not', 'it', 'a', 'and', 'to', 'you', 'the', 'i']
# top 200
COMMON_WORDS1_2 = ['had', 'call', 'littl', 'everi', 'good', 'did', 'some', 'said', 'then', 
'hand', 'his', 'about', 'an', 'over', 'wanna', 'around', 'still', 'whi', 'mind', 'wo', 'man', 
'keep', 'or', 'too', 'give', 'tri', 'who', 'tell', 'live', 'through', 'right', 'caus', 'more',
 'gonna', 'been', 'world', 'night', 'thing', 'where', 'think', 'eye', 'need', 'how', 'look', 
 'by', 'her', 'could', 'here', 'yeah', 'day', 'back', 'life', 'at', 'as', 'he', 'ca', 'would',
  'make', 'let', 'way', 'say', 'take', 'want', 'got', 'feel', 'from', 'if', 'was', 'never', 
  'see', 'she', 'oh', 'they', 'get', 'down', 'one', 'out', 'come', 'there', 'go', 'can', 'up',
   'time', 'now', 'when', 'like', 'just', 'with', 'what', 'love', 'but', 'no', 'know', 'this',
    'so', 'have', 'be', 'for', 'will', 'all', 'on', 'am', 'we', 'are', 'that', 'do', 'your', 
    'my', 'of', 'is', 'in', 'me', 'not', 'it', 'a', 'and', 'to', 'you', 'the', 'i']
# words that appeared in the top 100 words of 5 genres
COMMON_WORDS2 = ['thing', 'babi', 'her', 'day', 'life', 'away', 'as', 'feel', 'she', 'oh']
# top 200
COMMON_WORDS2_2 = ['realli', 'new', 'than', 'stay', 'cri', 'much', 'better', 'hear', 
'them', 'walk', 'believ', 'ever', 'hold', 'girl', 'leav', 'run', 'long', 'find',
 'well', 'us', 'onli', 'again', 'babi', 'heart', 'our', 'away']
# words that appeared in the top 100 words of 4 genres
COMMON_WORDS3 = ['right', 'caus', 'gonna', 'been', 'night', 'heart', 'by', 'could']
# words that appeared in the top 100 words of any genre
ALL_COMMON_WORDS = ['tell', 'live', 'die', 'through', 'again', 'right', 'caus', 'more', 'gonna', 
'been', 'world', 'night', 'thing', 'where', 'think', 'babi', 'eye', 'need', 'heart', 'how', 'our',
 'look', 'by', 'her', 'could', 'here', 'yeah', 'day', 'back', 'life', 'at', 'away', 'as', 'he', 'ca',
  'would', 'make', 'let', 'way', 'say', 'take', 'want', 'got', 'feel', 'from', 'if', 'was', 'never', 
  'see', 'she', 'oh', 'they', 'get', 'down', 'one', 'out', 'come', 'there', 'go', 'can', 'up', 'time',
   'now', 'when', 'like', 'just', 'with', 'what', 'love', 'but', 'no', 'know', 'this', 'so', 'have', 
   'be', 'for', 'will', 'all', 'on', 'am', 'we', 'are', 'that', 'do', 'your', 'my', 'of', 'is', 'in',
    'me', 'not', 'it', 'a', 'and', 'to', 'you', 'the', 'i',  'about', 'girl', 'und', 
    'an', 'off', 'keep', 'them', 'then', 'em', 'some', 'wanna', 'bitch', 'ich', 'or', 'who', 'de', 
    'fuck', 'la', 'yo', 'man', 'shit', 'ya', 'nigga', 'shake', 'too', 'did', 'wo', 'home', 'yes', 
    'said', 'blue', 'woman', 'long', 'good', 'lord', 'well', 'sing', 'believ', 'onli', 'him', 'us', 
    'jesus', 'chorus', 'give', 'his', 'god', 'had', 'still', 'old', 'littl', 'find', 'whi', 'que', 'dream']


"""
Training and testing data all came from 
http://millionsongdataset.com/musixmatch/
"""

class Track(object):
    """
    Represents a music track
    attributes
    track_number -> string
    genre -> string
    lyrics -> dictionary
    word_count -> int (total number of words in song)
    guessed_genre -> string

    """
    genre_dict = {"Blues":0, "Pop_Rock":0, "Rap":0, "Religious":0, "Country":0, "Jazz":0}

    def __init__(self, tracknum, genre, lyr, numberwords):
        self.track_number = tracknum
        self.genre = genre
        self.lyrics = lyr 
        self.word_count = numberwords
        self.guessed_genre = None
        self.guessed_list = None

    def setGuessedGenre(self, fg):
        self.guessed_genre = fg
        Track.genre_dict[fg] = Track.genre_dict[fg] + 1

    def genreMatch(self):
        """
        Returns True if the guessed genre matches the real genre
        """
        return self.genre == self.guessed_genre

    def setGuessedList(self, l):
        self.guessed_list = l
    
    def findAnswer(self):
        for x in range(len(self.guessed_list)):
            if self.guessed_list[x][0] == self.genre:
                return (x+1)
    def gMatch(self):
        return self.genre == self.guessed_list[1][0]

    @classmethod
    def getDict(cls):
        return Track.genre_dict


def mysort(l):
    """
    Takes in a list of tuples (genre, number), and sorts the tuples in 
    descending order
    """
    d = {}
    second = []
    for x in l:
        d[x[1]] = x
        second.append(x[1])
    second.sort()
    third = []
    i = len(second) -1 
    while i >= 0:
        third.append(d[second[i]])
        i -= 1
    return third

def reviewList(l):
    """
    Takes in a list of tuples (genre, number) 

    Returns list with any tuple removed where the number is less than 10% of the total numbers
    """
    mini = l[0][1]
    for t in l:
        mini = min(mini, t[1])
    newlis = []
    for t in l:
        nt = (t[0], t[1] - mini)
        newlis.append(nt)
    sum = 0
    for t in newlis:
        sum += t[1]
    flist = []
    five = sum * .2
    for t in newlis:
        if t[1] > five:
            flist.append(t)
    
    return mysort(flist)


def classify(bigN, nL, word_dict, song_dict):
    """
    Returns the summation
    log(Nl/N) + sum (log(njkjl/Nl))
    
    bigN = number of words in the dataset
    nL = number of words in the class we're checking
    word_dict = dictionary of word count for the class we're checking
    song_dict = dictionary of words in the song
    """
    sum =  math.log(nL/bigN)
    for k in song_dict.keys(): 
        try:
            njk = int(word_dict[k])
        except:
            njk = 0
        song_njk = int(song_dict[k])
        if njk == 0:
            to_add = 0
        elif song_njk == 0:
            to_add = 0
        else:
            to_add = (song_njk*song_njk)* (math.log(njk/nL))
  
        sum += to_add
    return sum 

def classify2(bigN, nL, word_dict, song_dict):
    """
    Uses the Naive Bayes formula, no smoothing, but with logs
    """
    x1 = math.log(nL/bigN)
    sum = 0
    for item in song_dict.keys():
        try:
            njkl = int(word_dict[item])
            song_njkl = int(song_dict[item])
        
            if njkl == 0:
                toadd = 0
            else:
                toadd = song_njkl*math.log(njkl/nL)
            sum += toadd
        except:
            pass
    return sum*x1


        
def classify3(bigN, nL, word_dict, song_dict):
    """
    Uses Naive Bayes formula, with Laplace smoothing and logs
    """

    x1 = math.log(nL/bigN)
    
    sum = 0
    for item in song_dict.keys():
        try:
            njkl = int(word_dict[item]) 
            song_njkl = song_dict[item]
           
            toadd =  song_njkl* math.log((njkl + 1)/(nL + 2))
            
            sum += toadd
            
        except:
            pass
            
    return sum*x1



def genrefy(song, dict_list, bigN):
    """
    Takes in a track (song) and a list of genre dictionaries (dict_list) and sets
    the guessed genre of the track to the genre it most closely matches
    """
    lyrics = song.lyrics
    lis = []
    maximum = classify(bigN, int(dict_list[0]['totalwords!']), dict_list[0], lyrics)
    curr_genre = dict_list[0]["GENRE!!"] 
    for d in dict_list:
        n = classify(bigN, int(d['totalwords!']), d, lyrics)
        lis.append((d["GENRE!!"], n))
        if n > maximum: 
            maximum = n
            curr_genre = d["GENRE!!"]
    song.setGuessedGenre(curr_genre)
    lis.sort()
    song.setGuessedList(lis)

def genrefy2(lyrics, dict_list, bigN):
    """
    Takes in the lyrics of a song (as a dictionary)
    Returns the genre the song matches
    """
    maximum = classify(bigN, int(dict_list[0]['totalwords!']), dict_list[0], lyrics)
    curr_genre = dict_list[0]["GENRE!!"] 
    mylist = []
    for d in dict_list:
        n = classify(bigN, int(d['totalwords!']), d, lyrics)
        mylist.append((d["GENRE!!"], n))
        if n > maximum: 
            maximum = n
            curr_genre = d["GENRE!!"]
    m = reviewList(mylist)
    return m




def readingTestData():
    """
    Reads the test data file and turns each track into a Track object
    Returns the list of track objects
    """    
    track_to_genre = readGenres()
    f = open(FILENAME_TEST, "r")
    track_list = []
    words_dict = {}
    words_list = [""]
    reading = True 
    for x in f.readlines():
        if x == "NEXT\n":
            # this is where we transition to loading into dataset 
            reading = False
            print("done reading words")
            print("starting reading tracks")
        elif reading:
            # reading words
            wlist = x.split(",")
            for y in wlist:
                words_list.append(y.strip())
                words_dict[y.strip()] = 0
        else:
            # loading tracks into the dataset
            # TRAAAAV128F421A322,4623710,1:6,2:4,3:2,4:2,5:5,6:3,
            try:
                wlist = x.split(",")
                tracknum = wlist[0]
                genre = track_to_genre[tracknum]
                d = {}
                numwords = 0
                for y in wlist:
                    pos = y.find(":")
                    if pos != -1:
                        wordnum = int(y[:pos])
                        wordcount = int(y[pos+1:].strip())
                        theword = words_list[wordnum]
                        d[theword] =  wordcount
                        numwords += wordcount
                thistrack = Track(tracknum, genre, d, numwords)
                track_list.append(thistrack)
            except:
                pass
    f.close()
    return track_list

def readGenres():
    """
    Returns track to genre dictionary
    """
    # create track to genre dictionary
    f = open(FILENAME_GENRE, "r")
    track_to_genre = {}
    for line in f.readlines():
        s = line.strip().split("\t")
        if s[1] in accepted_genres:
            track_to_genre[s[0]] = s[1]
    f.close()
    return track_to_genre


def readingfiles():
    """
    Reads in .txt files and turns them into dictionaries,
    where each key value pair is on a separate line,
    written key:value
    """
    track_to_genre = readGenres()
    print("created genre dictionary")
    # now creating all the tracks 
    f= open(FILENAME_LYRICS,"r")
    reading = True
    words_list = [""]
    words_dict = {}
    #track_list = []
    pop_dict = {}
    rap_dict = {}
    blues_dict = {}
    religious_dict = {}
    country_dict = {}
    jazz_dict = {}

    print("starting reading lyrics")
    total_word_count = 0
    for x in f.readlines():
        if x == "NEXT\n":
            # this is where we transition to loading into dataset 
            reading = False
            pop_dict = words_dict.copy()
            rap_dict = words_dict.copy()
            blues_dict = words_dict.copy()
            religious_dict = words_dict.copy()
            country_dict = words_dict.copy()
            jazz_dict = words_dict.copy()
            pop_dict["totalwords!"] = 0
            pop_dict["GENRE!!"] = "Pop_Rock"
            rap_dict["totalwords!"] = 0
            rap_dict["GENRE!!"] = "Rap"
            blues_dict["totalwords!"] = 0
            blues_dict["GENRE!!"] = "Blues"
            religious_dict["totalwords!"] = 0
            religious_dict["GENRE!!"] = "Religious"
            country_dict["totalwords!"] = 0
            country_dict["GENRE!!"] = "Country"
            jazz_dict["totalwords!"] = 0 
            jazz_dict["GENRE!!"] = "Jazz"
            print("done reading words")
            print("starting reading tracks")
        elif reading:
            # reading words
            wlist = x.split(",")
            for y in wlist:
                words_list.append(y)
                words_dict[y] = 0
        else:
            # loading tracks into the dataset
            # TRAAAAV128F421A322,4623710,1:6,2:4,3:2,4:2,5:5,6:3,
            try:
                wlist = x.split(",")
                tracknum = wlist[0]
                genre = track_to_genre[tracknum]
                gdict = [pop_dict, rap_dict, blues_dict, religious_dict, country_dict, jazz_dict]
                gpos = accepted_genres.index(genre)
                operating_dict = gdict[gpos]
                #d = words_dict.copy()
                #numwords = 0
                for y in wlist:
                    pos = y.find(":")
                    if pos != -1:
                        wordnum = int(y[:pos])
                        wordcount = int(y[pos+1:])
                        theword = words_list[wordnum]
                        #d[theword] =  wordcount
                        #numwords += wordcount
                        operating_dict[theword] += wordcount
                        operating_dict["totalwords!"] += wordcount
                        total_word_count += wordcount
                #thistrack = Track(tracknum, genre, d, numwords)
                #track_list.append(thistrack)
            except:
                pass
    f.close()
    print("done creating tracks")
    #print("Found " + str(len(track_list)) + " acceptable songs") ~98000
    gdict = [pop_dict, rap_dict, blues_dict, religious_dict, country_dict, jazz_dict]
    return gdict, total_word_count

def createFile(item, filename):
    """
    Takes every key value pair in item and writes them to a text
    file such that each pair is on a separate line, written
    key:value, in no particular order
    item is a dictionary
    filename is a string containing the name of the file to create
    """
    f = open(filename, "w+")
    for k in item.keys():
        value = item[k]
        f.write(str(k) + ":" + str(value)+"\n")
    f.close()

def createDictFiles(gdict):
    """
    Creates a .txt file for every dictionary in gdict
    gdict is a list of dictionaries
    """
    print("taking a quick break to create some files")
    for x in range(6):
        createFile(gdict[x], FILENAMES[x])
    print("okie they created!!")

def readInDicts():
    """
    Reads in the dictionaries from the files created
    """
    pop_dict = {}
    rap_dict = {}
    blues_dict = {}
    religious_dict = {}
    country_dict = {}
    jazz_dict = {} 
    gdict = [pop_dict, rap_dict, blues_dict, religious_dict, country_dict, jazz_dict]
    total_word_count = 0
    for x in range(6):
        f = open("./words/"+FILENAMES[x], "r")
        for y in f.readlines():
            y = y.strip()
            try:
                pos = y.find(":")
                key = y[:pos]
                value = y[pos+1:].strip()
                gdict[x][key] = value
            except:
                pass
        total_word_count += int(gdict[x]["totalwords!"])
    return gdict, total_word_count

def reduceDicts(gdict):
    """
    For every key, finds the lowest value corresponding to that key across 
    all the dictionaries and subtracts that number from every value
    corresponding to that specific key across all the dictionaries. 
    """
    modified = 0
    dictionary = gdict[0]
    for key in dictionary.keys():
        if key != "totalwords!" and key != "GENRE!!":
            min = int(dictionary[key])
            for d in gdict:
                val = int(d[key])
                if val < min:
                    min = val 
            if min > 0:
                modified += 1
                for d in gdict:
                    d[key] = int(d[key]) - min 


 
def textToDict(text):
    """
    Where text is a string
    Returns a dictionary mapping words to the number of times they occur. 
    The keys in the dictionary are all stemmed words
    """
    text = text.lower()
    contractions = [("'m ", " am "), ("'re ", " are "), ("'ve ", " have "), ("'d ", " would "), (" he's ", " he is "),
    (" she's ", " she is "), (" it's ", " it is "), (" ain't ", " is not "), ("n't ", " not "), ("'s ", " "), ("\r", " "), ("\n", " ")]
    punctuation = [',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')', '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*']
    for x in contractions:
        text = text.replace(x[0], x[1])
    for x in punctuation:
        text = text.replace(x, "")
    wordlist = text.split(" ")
    worddict = {}
    for x in wordlist:
        if x != "":
            word = stem(x)
            try:
                worddict[word] += 1
            except:
                worddict[word] = 1
            
               
    return worddict 

def classify_lyrics(text):
    """
    Takes in a string of song lyrics and returns a list of 
    genres sorted in order of what genre the song is likely to be.
    """
    lyrics = textToDict(text)
    gdict, total_word_count = readInDicts()
    genre = genrefy2(lyrics, gdict, total_word_count)
    return genre


def main():
    """
    Runs test suite
    """
    start = time.time()
    # ["Pop_Rock", "Rap", "Blues", "Religious", "Country", "Jazz"]
    # list of dictionaries, bigN
    #gdict, total_word_count = readingfiles()
    gdict, total_word_count = readInDicts()
    print("reading test data")
    test_list = readingTestData()
    
    #createDictFiles(gdict)
    
    # assigns guesses for all data
    print("guessing")
    #reduceDicts(gdict)
    for item in test_list:
        genrefy(item, gdict, total_word_count)
    # checks how well we did on each guess
    print("so how'd we do???")
    totalTypes = [0, 0, 0, 0, 0, 0]
    totalRight = [0, 0, 0, 0, 0, 0]
    total_words = 0
    total_correct = 0
    for item in test_list:
        total_words += 1
        pos = accepted_genres.index(item.genre)
        totalTypes[pos] += 1
        if item.genreMatch():
            totalRight[pos] += 1
            total_correct += 1
    total_percent = 0
    for g in range(len(accepted_genres)):
        if totalTypes[g] != 0:
            percent = totalRight[g]/totalTypes[g]
            total_percent += percent 
            print(accepted_genres[g] + " was " + str(percent) + " percent accurate")
        else:
            print("Didn't have any examples of " + accepted_genres[g])
    
    print("The overall correctness was " + str(total_correct/total_words) + " percent correct")
    print("The average correctness was " + str(total_percent/6) + " percent correct")
    print("To see what we're guessing: ")
    print(Track.getDict())
    end = time.time()
    print("\n\nIt took soooooo long to run this program!!!!!")
    print(time.strftime("%H:%M:%S", time.gmtime(end-start)))


                
def countgenres():
    """
    Counts the number of songs guessed by the algorithm for each genre
    """
    f = open(FILENAME_GENRE, "r")
    track_to_genre = {}
    for line in f.readlines():
        s = line.strip().split("\t")
        try:
            track_to_genre[s[1]] = track_to_genre[s[1]] + 1
        except:
            track_to_genre[s[1]] = 1 
    f.close()
    for x in track_to_genre.keys():
        print(x + " " + str(track_to_genre[x]))


def newRead():
    """
    Reads in training data, counts the number of songs with each word in each genre
    """
    track_to_genre = readGenres()
    print("created genre dictionary")
    # now creating all the tracks 
    f= open(FILENAME_LYRICS,"r")
    reading = True
    words_list = [""]
    words_dict = {}
    #track_list = []
    pop_dict = {}
    rap_dict = {}
    blues_dict = {}
    religious_dict = {}
    country_dict = {}
    jazz_dict = {}

    print("starting reading lyrics")
    total_word_count = 0
    for x in f.readlines():
        if x == "NEXT\n":
            # this is where we transition to loading into dataset 
            reading = False
            pop_dict = words_dict.copy()
            rap_dict = words_dict.copy()
            blues_dict = words_dict.copy()
            religious_dict = words_dict.copy()
            country_dict = words_dict.copy()
            jazz_dict = words_dict.copy()
            pop_dict["totalwords!"] = 0
            pop_dict["GENRE!!"] = "Pop_Rock"
            rap_dict["totalwords!"] = 0
            rap_dict["GENRE!!"] = "Rap"
            blues_dict["totalwords!"] = 0
            blues_dict["GENRE!!"] = "Blues"
            religious_dict["totalwords!"] = 0
            religious_dict["GENRE!!"] = "Religious"
            country_dict["totalwords!"] = 0
            country_dict["GENRE!!"] = "Country"
            jazz_dict["totalwords!"] = 0 
            jazz_dict["GENRE!!"] = "Jazz"
            print("done reading words")
            print("starting reading tracks")
        elif reading:
            # reading words
            wlist = x.split(",")
            for y in wlist:
                words_list.append(y)
                words_dict[y] = 0
        else:
            # loading tracks into the dataset
            # TRAAAAV128F421A322,4623710,1:6,2:4,3:2,4:2,5:5,6:3,
            try:
                wlist = x.strip().split(",")
                tracknum = wlist[0]
                genre = track_to_genre[tracknum]
                gdict = [pop_dict, rap_dict, blues_dict, religious_dict, country_dict, jazz_dict]
                gpos = accepted_genres.index(genre)
                operating_dict = gdict[gpos]
                operating_dict["totalwords!"] += 1 # using totalwords as the total number of songs of that genre
                total_word_count += 1 # where total word count is the total number of songs in the genre
                for y in wlist:
                    pos = y.find(":")
                    if pos != -1:
                        wordnum = int(y[:pos])
                        wordcount = int(y[pos+1:])
                        theword = words_list[wordnum]
                        if wordcount > 0:
                            operating_dict[theword] += 1
                            
                            
                
            except:
                pass
    f.close()
    print("done creating tracks")
    #print("Found " + str(len(track_list)) + " acceptable songs") ~98000
    gdict = [pop_dict, rap_dict, blues_dict, religious_dict, country_dict, jazz_dict]
    return gdict, total_word_count


#main() 