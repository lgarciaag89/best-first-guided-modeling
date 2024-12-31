package tomocomd.utils;

import static org.junit.jupiter.api.Assertions.*;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import org.junit.jupiter.api.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

public class PearsonCorrelationBetweenAttributesTest {

  @Test
  public void testComputeCorrelationMatrix() {
    Instances data = getData();
    TriangularMatrixAsVector result =
        PearsonCorrelationBetweenAttributes.computeCorrelationMatrix(data);
    double[] corr = result.getVector();
    assertEquals(getCorrExpected().length, corr.length);
    for (int i = 0; i < corr.length; i++) {
      assertEquals(getCorrExpected()[i], corr[i], 1e-15);
    }
  }

  double[] getCorrExpected() {
    return new double[] {
      1.0,
      -0.14146919145389775,
      1.0,
      -0.4999408583713647,
      0.9028124404816812,
      1.0,
      0.07256281304120243,
      -0.9728373537526354,
      -0.8998988020761235,
      1.0,
      -0.5629260024996691,
      0.36708439870635745,
      0.3640058203840977,
      -0.1514270306331352,
      1.0
    };
  }

  Instances getData() {
    ArrayList<Attribute> attInfo = new ArrayList<>();
    attInfo.add(new Attribute("att1"));
    attInfo.add(new Attribute("att2"));
    attInfo.add(new Attribute("att3"));
    attInfo.add(new Attribute("att4"));
    attInfo.add(new Attribute("att5"));

    Instances dataset = new Instances("dataset", attInfo, 4);
    dataset.add(new DenseInstance(1.0, new double[] {0.38, 63.76, 184.58, 204483.85, 201287.82}));
    dataset.add(new DenseInstance(1.0, new double[] {0.37, 63.96, 184.58, 204509.22, 206035.95}));
    dataset.add(new DenseInstance(1.0, new double[] {0.57, 63.88, 166.51, 203881.59, 201215.80}));
    dataset.add(new DenseInstance(1.0, new double[] {0.41, 64.59, 223.50, 197788.30, 203764.31}));

    return dataset;
  }
}
