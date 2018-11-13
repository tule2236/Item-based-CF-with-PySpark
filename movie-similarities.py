'''
Map input rating to (userID,(movieID, rating))
Compute the movie pairs of the same user: self-join
	(userID, ( (movie1, rating1), (movie2, rating2) ) )
	(userID, ( (movie1, movie2), (rating1, rating2) ) )
	Filter out duplicate movie pairs
Make the movie pair (movie1, movie2) the key
	Compute similarity b/w any movie pairs # use "groupByKey()"
Sort, save, and display the results
'''
import sys
from pyspark import SparkConf, SparkContext
from math import sqrt

def loadMovieNames():
    movieNames = {}
    with open("Data/ml-100k/u.item", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

def checkDuplicates(pairs):
	(movie1, rating1) = pairs[1][0]
	(movie2, rating2) = pairs[1][1]
	return movie1 < movie2

def makePairs(userRating):
	(movie1, rating1) = userRating[1][0]
	(movie2, rating2) = userRating[1][1]
	return ( (movie1, movie2), (rating1, rating2) ) 

def computeCosineSimilarity(ratingPair): #(rating1, rating2)
	numPairs = 0
	sum_xx = sum_yy = sum_xy = 0
	for ratingX, ratingY in ratingPair:
		sum_xx += ratingX * ratingX
		sum_yy += ratingY * ratingY
		sum_xy += ratingX * ratingY
		numPairs += 1
	cosine = 0
	denominator = sqrt(sum_xx) * sqrt(sum_yy)
	if denominator:
		cosine = sum_xy / float(denominator)
	return (cosine, numPairs)

conf = SparkConf().setMaster("local[*]").setAppName("MovieSimilarities")
sc = SparkContext(conf = conf)
nameDict = loadMovieNames()
# Data: (UserID, MovieID, Rating, TimeStamp)
lines = sc.textFile("Data/ml-100k/u.data")
rating = lines.map(lambda x: x.split()).map(lambda x: (int(x[0]), (int(x[1]), float(x[2]))) ) 
# userID => ((movieID, rating), (movieID, rating))
mapRating = rating.join(rating) #self join to find movie pairs of the same user
filterRating = mapRating.filter(checkDuplicates)
# key by (movie1, movie2) pairs.
pairs = filterRating.map(makePairs)
# (movie1, movie2) => (rating1, rating2)
groupedPairs = pairs.groupByKey()
# (movie1, movie2) = > (rating1, rating2), (rating1, rating2) 
similarPairs = groupedPairs.mapValues(computeCosineSimilarity).cache() 
#userID =>  (movie1, movie2), (cosine, numPairs) )

if len(sys.argv) > 1: # user input movieID to find similar group from
	movieID = int(sys.argv[1])
	scoreThreshold = 0.97 #threshold for cosine similarity
	coOccurenceThreshold = 50 #threshold for number of similar neighbors
	filterResults = similarPairs.filter(lambda x: ((x[0][0] == movieID) or (x[0][1] == movieID)) \
		and ((x[1][0]) >= scoreThreshold) and (x[1][1] >= coOccurenceThreshold) )
	# Sort the score
	results = filterResults.map(lambda x: (x[1], x[0])).sortByKey(ascending = False).take(10)
	print("Top 10 similar movies for movie " + nameDict[movieID])
	for result in results:
		(sim, pair) = result # (cosine, numPairs),(movie1, movie2)
		# Display the similarity result that isn't the movie we're looking at
		similarMovieID = pair[0]
		if (similarMovieID == movieID):
		    similarMovieID = pair[1]
		print(nameDict[similarMovieID] + "\tscore: " + str(sim[0]) + "\tstrength: " + str(sim[1]))




















