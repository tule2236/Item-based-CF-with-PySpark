# Find the top 10 most popular movies
 # Data: (UserID, MovieID, Rating, TimeStamp)
from pyspark.sql import SparkSession
from pyspark.sql import Row


def loadMovieName():
	movieNames = {}
	with open('Data/ml-100k/u.item', encoding="ISO-8859-1") as f:
		for line in f:
			fields = line.split('|')
			movieNames[int(fields[0])] = fields[1]
	return movieNames

spark = SparkSession.builder.appName("PopularMovies").getOrCreate()
movieLines = spark.sparkContext.textFile('Data/ml-100k/u.data')
nameDict = loadMovieName()
# Convert RDD to DataFrame
movies = movieLines.map(lambda x: Row( movieID = int(x.split()[1]) ))
movieDataset = spark.createDataFrame(movies)
topMovieIDs = movieDataset.groupBy("movieID").count().orderBy("count", ascending=False).cache()
topMovieIDs.show()
top10 = topMovieIDs.take(10)
for topID in top10:
	print("%s: %d" % (nameDict[topID[0]], topID[1]))
spark.stop()