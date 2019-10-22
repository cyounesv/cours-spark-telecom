package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel, IDF, RegexTokenizer, StopWordsRemover, VectorIndexer}
//spark.ml.classification,
import org.apache.spark.ml.evaluation
import org.apache.spark.ml.tuning
import  org.apache.spark.ml.Pipeline


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")
    val df = spark.read.parquet("/home/cyounes/Documents/cours-spark-telecom/data/prepared_trainingset")
    df.show()

    df.groupBy("country2").count.show(100)
    //df.groupBy("currency2").count.show(100)
 /*
    //STAGE 1
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")
    val wordsData = tokenizer.transform(df)
//wordsData.show(5)
    println("hello world ! from Trainer 2")
    //STAGE 2
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("removed")
      .setStopWords(Array("the","a","http","i","me","to","what","in","rt"))

    val cleanWordsData = remover.transform(wordsData)
  //  cleanWordsData.show()
    //STAGE 3
    println("hello world ! from Trainer 3")
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("tokens")
      .setOutputCol("features")
      .setVocabSize(100)
      .fit(cleanWordsData)  //

    val dfTest = cvModel.transform(cleanWordsData)
  //  dfTest.show(5)
    println("hello world ! from Trainer 4")
    //STAGE 4 :  Implémenter la partie IDF avec en output une colonne tfidf.
    val idf = new IDF()
      .setInputCol("features")
      .setOutputCol("tfidf")

    val idfModel = idf.fit(dfTest)
    val rescaledData = idfModel.transform(dfTest)
    rescaledData.select("label", "tfidf").show()
*/
/*
    newDataSet
      .select("removed") // if you're interested only in "clean" text
      .map(row => row.getSeq[String](0).mkString(" ")) // make Array[String] into String
      .write.text("/path/to/SherlockHolmsWithoutStopWords.txt")


    val splits = cleanData.randomSplit(Array(0.9, 0.1))     // splitting training and test data
    val (trainingData, testData) = (splits(0), splits(1))

    val lr = new LogisticRegression()                       // use  of Logistic regression Classifier
      .setElasticNetParam(1.0)                              // L1 penalisation --> LASSO
      .setStandardization(true)
      .setFitIntercept(true)
      .setTol(1.0e-5)
      .setMaxIter(300)

*/


    //STAGE 5
    //There are 11 countries and 9 currencies (number found by groupby.count query)
    // Est ce que j'ai besoin de garder le pays/la currency asso a chaque index?

    val indexer = new VectorIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setMaxCategories(11)

    val dfCategorical = indexer.transform(df)

    //STAGE 6
    val indexer2 = new VectorIndexer()
      .setInputCol("currency3")
      .setOutputCol("currency_indexed")
      .setMaxCategories(9)


    val dfCategorical2 = indexer.transform(dfCategorical)
    //STAGE 7 & 8

  }
}
