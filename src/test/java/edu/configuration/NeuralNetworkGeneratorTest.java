package edu.configuration;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static edu.configuration.DataSetLoader.getDataSet;
import static edu.configuration.NeuralNetworkGenerator.getNeuralNetConfiguration;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkGeneratorTest {

    @Test
    void getNeuralNetConfigurationTest() throws IOException, InterruptedException {
        MultiLayerConfiguration configuration = getNeuralNetConfiguration();

        System.out.println("Loading data set");
        Path path = Paths.get("src", "test", "resources", "data", "regularized", "dataR.txt");
        DataSet loadedDataSet = getDataSet(path.toFile());
        loadedDataSet.normalize();
        SplitTestAndTrain splitTestAndTrain = loadedDataSet.splitTestAndTrain(0.65);

        DataSet testDataSet = splitTestAndTrain.getTest();
        loadedDataSet = splitTestAndTrain.getTrain();
        List<DataSet> dataSets = loadedDataSet.batchBy(10);
        MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        multiLayerNetwork.init();

        int epochs = 200;
        for (int i = 1; i <= epochs; i++) {
            System.out.println("Epoch: " + i + "/" + epochs);
            for (DataSet dataSet : dataSets) {
                multiLayerNetwork.fit(dataSet);
            }
        }

        Evaluation evaluation = new Evaluation(2);
        evaluation.eval(testDataSet.getLabels(), multiLayerNetwork.output(testDataSet.getFeatures()));

        assertTrue(evaluation.accuracy() > 0.95d, evaluation.accuracy() + " is to small");
        assertTrue(evaluation.precision() > 0.91d, evaluation.precision() + " is to small");
        assertTrue(evaluation.recall() > 0.95d, evaluation.recall() + " is to small");
        assertTrue(evaluation.f1() > 0.95d, evaluation.f1() + " is to small");
    }
}