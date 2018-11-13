# Data: (UserID, MovieID, Rating, TimeStamp)
# Goal: count how many time each movie occur in the dataset => find most popular movies
from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local").setAppName("PopularMovies")
sc = SparkContext(conf = conf)

def loadMovieName():
	movieName = {}
	with open("ml-100k/u.item") as f:
		for line in f:
			fields = line.split("|")
			movieName[int(fields[0])] = fields[1] #{movieID: movieName}
	return movieName

#save an object into the executor (w/t broadcast) so it's available across all executors
nameDict = sc.broadcast(loadMovieName())

lines = sc.textFile("ml-100k/u.data")
movie = lines.map(lambda x: (int(x.split()[1]), 1))
movieCounts = movie.reduceByKey(lambda x,y:x+y)
flipped = movieCounts.map(lambda x: (x[1], x[0]))
sortedMovieCounts = flipped.sortByKey() #(freq,movie)
# broadcast has to retrieve within RDD
sortedMovieWithName = sortedMovieCounts.map(lambda movie: (nameDict.value[movie[1]], movie[0]))
results = sortedMovieWithName.collect() #(freq, movieID)
for result in results:
	# print((nameDict[result[1]], result[0]))
	print(result)
