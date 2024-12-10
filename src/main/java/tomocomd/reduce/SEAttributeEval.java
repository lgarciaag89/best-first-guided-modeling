package tomocomd.reduce;

import tomocomd.utils.SEAttribute;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

public class SEAttributeEval extends ASEvaluation implements AttributeEvaluator {

  private double[] m_weights;

  @Override
  public double evaluateAttribute(int attribute) throws Exception {
    return m_weights[attribute];
  }

  @Override
  public void buildEvaluator(Instances data) throws Exception {
    m_weights = new double[data.numAttributes()];

    for (int i = 0; i < data.numAttributes(); i++) {
      m_weights[i] = SEAttribute.computeEntropy(data.attributeToDoubleArray(i));
    }
  }
}
