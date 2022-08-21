package edu.configuration;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.learning.config.Nesterovs;

import static org.nd4j.linalg.activations.Activation.RELU;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.SQUARED_LOSS;

public class NeuralNetworkGenerator {
    public static MultiLayerConfiguration getNeuralNetConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .seed(12).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .biasInit(0.5).updater(new Nesterovs(0.00009, 0.0005))
                .l2(0.000006)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(100).nOut(50).hasBias(true).activation(RELU).build())
                .layer(1, new DenseLayer.Builder().nIn(50).nOut(50).hasBias(true).activation(RELU).build())
                .layer(2, new DenseLayer.Builder().nIn(50).nOut(50).hasBias(true).activation(RELU).build())
                .layer(3, new OutputLayer.Builder().nIn(50).nOut(1).hasBias(true).lossFunction(SQUARED_LOSS).activation(RELU).build())
                .build();
    }
}
