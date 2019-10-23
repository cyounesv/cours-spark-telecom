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
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

    val cleanWordsData = remover.transform(wordsData)
 //   val df2 = cleanWordsData.select("tokens", "removed", "text")
   //   df2.show(10, false)

    //STAGE 3 : TF(t,d)
    println("hello world ! from Trainer 3")
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("removed")
      .setOutputCol("features")
      //.setVocabSize(2000)
      .fit(cleanWordsData)  //

    val dfTf = cvModel.transform(cleanWordsData)
    //val df1 = dfTf.select("removed", "features")
    //df1.show(10, false )
    println("hello world ! from Trainer 4")

    //STAGE 4 :  Implémenter la partie IDF avec en output une colonne tfidf.
    val idf = new IDF()
      .setInputCol("features")
      .setOutputCol("tfidf")

    val idfModel = idf.fit(dfTf)
    val rescaledData = idfModel.transform(dfTf)
   // rescaledData.select( "tfidf", "features").show(truncate = false)

//STAGE 5
//There are 11 countries and 9 currencies (number found by groupby.count query)

    val cvModel2: CountVectorizerModel = new CountVectorizer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setVocabSize(11)
      .fit(rescaledData)

    cvModel2.transform(rescaledData).show(false)

   /* val indexer = new VectorIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setMaxCategories(11)

    val indexerModel = indexer.fit(rescaledData)

    val categoricalFeatures: Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${categoricalFeatures.size} " +
      s"categorical features: ${categoricalFeatures.mkString(", ")}")

    // Create new column "indexed" with categorical values transformed to indices
    val indexedData = indexerModel.transform(rescaledData)
    indexedData.show()*/


  /*  //STAGE 6
    val indexer2 = new VectorIndexer()
      .setInputCol("currency3")
      .setOutputCol("currency_indexed")
      .setMaxCategories(9)


    val dfCategorical2 = indexer.transform(dfCategorical)*/


    //STAGE 7 & 8
    //Transformer ces deux catégories avec un "one-hot encoder" en créant les colonnes country_onehot et currency_onehot.
    //Get distinct countries & Get distinct currenci
 //   val pays = indexedData.groupBy("country_2")

   // print(pays)


   /* s = 'black holes'
    print(s)
    poss = 'abcdefghijklmnopqrstuvwxyz '
    char_to_int = dict((j, i) for i, j in enumerate(poss))
    int_to_char = dict((i, j) for i, j in enumerate(poss))
    integer_encoded = [char_to_int[i] for i in s]
    print(integer_encoded)

    onehot_encoded = list()
    for i in integer_encoded:
      t=[0 for j in range(len(poss))]
    t[i]=1
    onehot_encoded.append(t)
    print(onehot_encoded)
    inverted = int_to_char[argmax(onehot_encoded[0])]
    print(inverted)*/


    /*


        val splits = cleanData.randomSplit(Array(0.9, 0.1))     // splitting training and test data
        val (trainingData, testData) = (splits(0), splits(1))



    */




  }
}
