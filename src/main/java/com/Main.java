package com;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Main {

    private static MultiLayerConfiguration getNeuralNetConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(12).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .biasInit(0.5).updater(new Nesterovs(0.00009, 0.0005))
                .l2(0.000006)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(100).nOut(50).hasBias(true).activation(Activation.RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(50).nOut(50).hasBias(true).activation(Activation.RELU).build())
                .layer(2, new DenseLayer.Builder().nIn(50).nOut(50).hasBias(true).activation(Activation.RELU).build())
                .layer(3, new OutputLayer.Builder().nIn(50).nOut(1).hasBias(true).lossFunction(LossFunctions.LossFunction.SQUARED_LOSS).activation(Activation.RELU).build())
                .build();
    }

    private static DataSet getDataSet(String fileName) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(new File(new ClassPathResource("data.txt").getURI())));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000, 100, 1);
        DataSet dataSet = dataSetIterator.next();
        dataSet.shuffle(42);
        return dataSet;
    }

    public static void main(String[] args) {

        MultiLayerConfiguration configuration = getNeuralNetConfiguration();


        System.out.println("Loading data set");
        try {
            DataSet dataSet = getDataSet("data.txt");
            SplitTestAndTrain splitTestAndTrain = dataSet.splitTestAndTrain(0.65);

            DataSet testDataSet = splitTestAndTrain.getTest();
            dataSet = splitTestAndTrain.getTrain();
            List<DataSet> dataSets = dataSet.batchBy(50);
            MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
            multiLayerNetwork.init();

            for (DataSet ds : dataSets) {
                for (int i = 0; i < 120; ++i) {
                    multiLayerNetwork.fit(ds);
                    System.out.println("training...");
                }
            }

            Evaluation evaluation = new Evaluation(2);
            evaluation.eval(testDataSet.getLabels(), multiLayerNetwork.output(testDataSet.getFeatures()));
            System.out.println(evaluation.stats());
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
