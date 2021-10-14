import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineStage
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._

object ML {
  def main(args: Array[String]): Unit = {
    val input_train = args(0)
    val input_test = args(1)
    val output = args(2)
    val spark = SparkSession.builder().getOrCreate()

    val train = spark.read.format("org.elasticsearch.spark.sql").option("header", "true").csv(input_train)
      .toDF("text", "label")

    val training = train
      .na.fill("", Seq("text"))
      .selectExpr("text",
      "cast(label as float) label")


    // Конфигурируем ML pipeline
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words");
    val hashingTF = new HashingTF().setNumFeatures(1000).setInputCol(tokenizer.getOutputCol).setOutputCol("features")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001)
    val pipeline = new Pipeline().setStages(Array[PipelineStage](tokenizer, hashingTF, lr))


    // Запустим последовательность трансформаций и обучение на наших данных
    val model = pipeline.fit(training)

    // Подготовим тестовый датасет
    val test = spark.read.option("header", "true").csv(input_test)
      .toDF("text", "label" )

    val testing = test
      .na.fill("", Seq("text"))
      .selectExpr("text",
        "cast(label as float) label")

    import org.apache.spark.sql.functions._
    model.transform(testing)
      .select("label", "text","probability", "prediction")
      .write.mode("overwrite").csv(output)


        spark.stop()
  }
}