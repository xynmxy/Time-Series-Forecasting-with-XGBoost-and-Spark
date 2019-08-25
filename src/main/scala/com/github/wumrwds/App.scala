package com.github.wumrwds


import ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.sql.{Column, SparkSession}

/**
  * Predicts time series with XGBoost-Spark.
  *
  * @author wumrwds
  */
object App {

    def main(args: Array[String]): Unit = {
//        if (args.length != 5) {
//            println("usage: program nworkers ")
//            sys.exit(1)
//        }
//
//        // get command line parameters
//        val Array(nWorkersStr, numThreadsStr) = args
//        val nWorkers = nWorkersStr.toInt

        // init spark session
        val sparkSession = SparkSession
                .builder()
                .master("local[4]")
                .appName("Xgboost_Time_Series_Forecasting")
                .getOrCreate()

        // read train & test data, then transfer to dataframe
        val trainRawDf = sparkSession.read.format("csv").option("header", "true").load("src/main/data/train.csv")
        val testRawDf = sparkSession.read.format("csv").option("header", "true").load("src/main/data/test.csv")

        // cast feature columns to double type
        def castColToDouble(column: Column) = column.cast(DoubleType)
        val featureDoubleCols = trainRawDf.columns.filter(colName => "f1".equals(colName) || "f2".equals(colName) || "f3".equals(colName))
                .map(featureColName => castColToDouble(col(featureColName)))
        val trainDoubleDf = trainRawDf.select((col("date") +: featureDoubleCols :+ castColToDouble(col("y"))): _*)
        val testDoubleDf = testRawDf.select((col("date") +: featureDoubleCols): _*)

        // set vector assembler for used features
        val vectorAssembler = new VectorAssembler().
                setInputCols(Array("f1", "f2", "f3")).
                setOutputCol("features")
        val trainDf = vectorAssembler.transform(trainDoubleDf)
        val testDf = vectorAssembler.transform(testDoubleDf)

        // set xgboost regressor
        val xgbRegressor = new XGBoostRegressor(
            Map("eta" -> 0.1f,
                "max_depth" -> 2,
                "objective" -> "reg:squarederror",
                "num_round" -> 100,
                "nworkers" -> 4,
                "use_external_memory" -> true
            )
        )
                .setFeaturesCol("features")
                .setLabelCol("y")

        // train
        val model = xgbRegressor.fit(trainDf)

        // predict
        val result = model.transform(testDf)

        // print result
        result.select(col("date"), col("prediction")).coalesce(1).show(100, false)

        sparkSession.stop()
    }

}
