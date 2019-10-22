package paristech

import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

import scala.math.BigDecimal
import org.apache.spark.sql
import org.apache.spark.sql.catalyst.expressions.DateDiff

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._
    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/
    println("\n")
    println("Hello World ! from Preprocessor")
    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("/home/cyounes/Documents/cours-spark-telecom/data/train_clean.csv")

    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))

    dfCasted.printSchema()

    //dfCasted
      //.select("goal", "backers_count", "final_status")
      //.describe()
     // .show

    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("country").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("currency").count.orderBy($"count".desc).show(100)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc).show(100)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc).show(100)
    dfCasted.select("goal", "final_status").show(30)
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc).show(50)

    // drop d'une colonne quasi entierement a false
    val df2: DataFrame = dfCasted.drop("disable_communication")
    // drop des données ne pouvant être connues qu'après la résolution de chaque évènement
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

   /* dfNoFutur.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)
      .show(50)*/

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      if (country.length != 2)
          null
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    val dfClean: DataFrame = dfCountry
      .filter($"final_status" === 1 || $"final_status" === 0)

    dfClean.groupBy($"final_status").count().orderBy($"count".desc).show()

    println("TOTO")


    val dfDays: DataFrame = dfClean
        .withColumn("days_campaign", datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
      .withColumn("hours_prepa", round(($"launched_at"-$"created_at")/3600.floatValue(),3))
        .drop("launched_at","deadline","created_at")

    val dfFinal: DataFrame = dfDays
      .withColumn("name", lower(col("name")))
      .withColumn("desc", lower(col("desc")))
      .withColumn("keywords", lower(col("keywords")))
      .withColumn("Text1", concat($"name", lit(" "), $"desc"))
      .withColumn("Text", concat($"Text1", lit(" "), $"keywords"))
      .drop($"Text1")

    //dfText.show(15)

    //Replacing null values
    val map = Map("days_campaign" -> -1, "hours_prepa" -> -1, "goal" -> -1, "country2" -> "unknown", "currency2" -> "unknown")
    dfFinal.na.fill(map)

    //dfFinal.show(50)

    dfFinal.write.parquet("/home/cyounes/Documents/cours-spark-telecom/Fichiers-parquets")
    println("\n")
  }
}


//1ere boucle avec les start de process
//2e boucles avec les wait for !!