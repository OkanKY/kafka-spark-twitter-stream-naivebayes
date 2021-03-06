package io.philipg.spark.consumer.service;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.regression.LabeledPoint;

import java.util.Arrays;

public class Train {

    public static final String DATA_PATH = "C:\\Users\\okaya\\Documents\\git\\okan\\kafka-spark-twitter-stream-demo\\spark-consumer\\20_newsgroup\\";

    public static void main(String []args){

        JavaSparkContext sc = new JavaSparkContext("local","NaiveBayes");
        JavaRDD<String> atheism = sc.textFile(DATA_PATH + "alt.atheism\\*");
        JavaRDD<String> graphics = sc.textFile(DATA_PATH + "comp.graphics\\*");
        JavaRDD<String> motorcycles = sc.textFile(DATA_PATH + "rec.motorcycles\\*");

        final HashingTF tf = new HashingTF(1000);

        JavaRDD<LabeledPoint> atheismExamples = atheism.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                return new LabeledPoint(5, tf.transform(Arrays.asList(line.split(" "))));
            }
        });
        JavaRDD<LabeledPoint> graphicsExamples = graphics.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                return new LabeledPoint(10, tf.transform(Arrays.asList(line.split(" "))));
            }
        });

        JavaRDD<LabeledPoint> motorcyclesExamples = motorcycles.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                return new LabeledPoint(15, tf.transform(Arrays.asList(line.split(" "))));
            }
        });

        JavaRDD<LabeledPoint> trainingData1 = atheismExamples.union(graphicsExamples);
        JavaRDD<LabeledPoint> trainingData2 = trainingData1.union(motorcyclesExamples);
        trainingData2.cache(); // Cache data since Logistic Regression is an iterative algorithm.


        final NaiveBayesModel model = NaiveBayes.train(trainingData2.rdd());


        String atheismWord = " The scenario you outline is reasonably "
                + "consistent, but all the evidence that I am familiar with not only does"
                + "not support it, but indicates something far different. The Earth, by"
                + "latest estimates, is about 4.6 billion years old, and has had life for"
                + "about 3.5 billion of those years. Humans have only been around for (at"
                + "most) about 200,000 years. But, the fossil evidence inidcates that life"
                + "has been changing and evolving, and, in fact, disease-ridden, long before"
                + "there were people. (Yes, there are fossils that show signs of disease..."
                + "mostly bone disorders, of course, but there are some.) Heck, not just"
                + "fossil evidence, but what we've been able to glean from genetic study shows"
                + "that disease has been around for a long, long time. If human sin was what"
                + "brought about disease (at least, indirectly, though necessarily) then"
                + "how could it exist before humans?";

        String compGraphicsWord =
                "I am looking to add voice input capability to a user interface I am " +
                        "developing on an HP730 (UNIX) workstation.  I would greatly appreciate " +
                        "information anyone would care to offer about voice input systems that are " +
                        "easily accessible from the UNIX environment. ";


        String motorcyclesWord =
                "When I got my knee rebuilt I got back on the street bike ASAP. I put " +
                        "the crutches on the rack and the passenger seat and they hung out back a " +
                        "LONG way. Just make sure they're tied down tight in front and no problemo. " ;

        Vector testAtheismWord = tf.transform(Arrays.asList(atheismWord.split(" ")));
        Vector testCompGraphicsWord = tf.transform(Arrays.asList(compGraphicsWord.split(" ")));
        Vector testMotorcyclesWord = tf.transform(Arrays.asList(motorcyclesWord.split(" ")));



        System.out.println("Prediction for atheismWord : " + model.predict(testAtheismWord));
        System.out.println("Prediction for compGrapWord : " + model.predict(testCompGraphicsWord));
        System.out.println("Prediction for motorcyclesWord : " + model.predict(testMotorcyclesWord));

    }
}
