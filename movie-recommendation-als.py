from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS, Rating
import sys

def loadMovieNames():
    movieNames = {}
    with open("Data/ml-100k/u.item", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1]
    return movieNames

conf = SparkConf().setMaster('local[*]').setAppName('MovieRecommendationALS')
sc = SparkContext(conf= conf)
sc.setCheckpointDir('checkpoint')
nameDict = loadMovieNames()
# Prepare training data
lines = sc.textFile("Data/ml-100k/u.data") # # Data: (UserID, MovieID, Rating, TimeStamp)
training_data = lines.map(lambda x: x.split()).map(lambda x: Rating( int(x[0]), int(x[1]), float(x[2]) ))
rank = 10
numIterations = 6
lr = ALS.train(training_data, rank, numIterations)

userID = int(sys.argv[1])
userRatings = training_data.filter(lambda x: x[0] == userID)
print("Rating for userID %d is: " % userID)
if (len(sys.argv) > 1):
	for userRating in userRatings.collect():
		print(nameDict[int(userRating[1])] + ' has ratings ' + str(userRating[2]))

	print("\nTop 10 movies recommendations are: ")
	recommendations = lr.recommendProducts(userID, 10)
	for recommendation in recommendations:
		print (nameDict[int(recommendation[1])] + \
        " score " + str(recommendation[2]))


