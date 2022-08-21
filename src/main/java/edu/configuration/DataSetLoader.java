package edu.configuration;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;

public class DataSetLoader {
    public static DataSet getDataSet(File dataSetFile) throws IOException, InterruptedException {
        RecordReader recordReader = new CSVRecordReader(0, ',');
        recordReader.initialize(new FileSplit(dataSetFile));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, 1000, 100, 1);
        DataSet dataSet = dataSetIterator.next();
        dataSet.shuffle(42);
        return dataSet;
    }
}
