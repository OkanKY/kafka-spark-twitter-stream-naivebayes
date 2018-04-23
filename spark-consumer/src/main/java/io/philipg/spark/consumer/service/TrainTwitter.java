package io.philipg.spark.consumer.service;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.util.Arrays;

public class TrainTwitter {

    public static final String DATA_PATH = "C:\\Users\\okaya\\Documents\\git\\okan\\kafka-spark-twitter-stream-demo\\spark-consumer\\spark-warehouse\\SentimentAnalysisDataset.csv";

    public static void main(String []args){

        JavaSparkContext sc = new JavaSparkContext("local","NaiveBayes");
        JavaRDD<String> allData = sc.textFile(DATA_PATH );

        final HashingTF tf = new HashingTF(1000);

        String header=allData.first();
        JavaRDD<String> data = allData.filter(x-> !x.contains(header));

//        val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
//        val training = splits(0)

        JavaRDD<String> []splits= data.randomSplit(new double[]{0.8, 0.2},11l);

        JavaRDD<String> training = splits[0];
        JavaRDD<String> test = splits[1];
        System.out.println("\n\n********* Train Set ********");

        JavaRDD<LabeledPoint> training_labeled = training.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
//                System.out.println(line);
                String[] words = line.split(",");
                return new LabeledPoint(getValue(words[1]),tf.transform(Arrays.asList(words[3].split(" "))));
            }
        });


        training_labeled.cache(); // Cache data since Logistic Regression is an iterative algorithm.

        final NaiveBayesModel model = NaiveBayes.train(training_labeled.rdd());

        JavaRDD<LabeledPoint> testining_labeled = training.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String[] words = line.split(",");
                return new LabeledPoint(getValue(words[1]),tf.transform(Arrays.asList(words[3].split(" "))));
            }
        });

        JavaRDD<Tuple2> predictionAndLabel = testining_labeled.map(new Function<LabeledPoint, Tuple2>() {
            public Tuple2 call(LabeledPoint line) {
//                String[] words = line.split(",");
                double prediction=model.predict(line.features());
                System.out.print("label: "+line.label());
                System.out.print("prediction: "+prediction);
                System.out.println();
                return new Tuple2(line.label(),prediction);
            }
        });

        predictionAndLabel.take(120).stream().forEach( x -> {
                System.out.println("---------------------------------------------------------------");
            System.out.println("Actual Label = " + ((Double)x._1 ==1 ? "positive" : "negative"));
            System.out.println("Predicted Label = " + ((Double)x._2==1 ? "positive" : "negative"));
            System.out.println("----------------------------------------------------------------\n\n");
         } );


        model.save(sc.sc(), "src/main/resources/myNaiveBayesModel");
        sc.stop();
        System.out.println("\n\n********* Stopped Spark Context succesfully, exiting ********");
    }
    public static double getValue(String value){
        try {
//            System.out.println();
//            System.out.print("value 1: "+words[1]+" ");
//            System.out.print("value 2: "+words[3]+" ");
            return Double.parseDouble(value);
        }catch (Exception e){
//            System.out.println("value 1: "+words[3]+" ");
            return 1.0;
        }
    }
}
