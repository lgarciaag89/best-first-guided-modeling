package tomocomd.searchmodels;

import java.util.*;
import java.util.stream.IntStream;
import tomocomd.CSVManage;
import tomocomd.ModelingException;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.SubsetEvaluator;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public abstract class ASearchModel extends ASEvaluation implements SubsetEvaluator {

  private static final String ERROR_MSG = "Error coping the classifier";

  protected String trainPath;

  protected Instances trainTotal;
  protected Instances testTotal;
  protected final String testPath;
  protected final String pathToSave;
  protected final int classAct;
  protected long modelId;
  protected final List<String> externalTestPath;
  protected final AbstractClassifier classifier;

  protected ASearchModel(
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    this.trainPath = trainPath;
    this.testPath = testPath;
    this.pathToSave = pathToSave;
    this.classAct = classAct;
    this.modelId = modelId;

    this.externalTestPath = new LinkedList<>(externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModel(
      Instances trainTotal,
      String trainPath,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    this.trainPath = trainPath;
    this.trainTotal = new Instances(trainTotal);
    this.testPath = testPath;
    this.pathToSave = pathToSave;
    this.classAct = classAct;
    this.modelId = modelId;

    this.externalTestPath = new LinkedList<>(externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  protected ASearchModel(
      Instances trainTotal,
      String trainPath,
      Instances testTotal,
      String testPath,
      String pathToSave,
      int classAct,
      long modelId,
      List<String> externalTestPath,
      AbstractClassifier classifier)
      throws ModelingException {
    this.trainPath = trainPath;
    this.trainTotal = new Instances(trainTotal);
    this.testPath = testPath;
    this.testTotal = new Instances(testTotal);
    this.pathToSave = pathToSave;
    this.classAct = classAct;
    this.modelId = modelId;

    this.externalTestPath = new LinkedList<>(externalTestPath);
    try {
      this.classifier = (AbstractClassifier) AbstractClassifier.makeCopy(classifier);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(ERROR_MSG, e);
    }
  }

  @Override
  public void buildEvaluator(Instances data) throws Exception {}

  @Override
  public double evaluateSubset(BitSet bitset) throws ModelingException {

    int[] idx =
        IntStream.concat(
                Arrays.stream(new int[] {0}), IntStream.of(bitset.stream().sorted().toArray()))
            .toArray();
    if (idx.length == 1) return 0;

    Instances train;
    Instances test = null;
    List<Instances> externalText = new LinkedList<>();
    if (idx.length < trainTotal.numAttributes()) {
      train = remove(trainTotal, idx);

      if (Objects.nonNull(testTotal)) test = remove(testTotal, idx);

      for (String extPath : externalTestPath) {
        Instances ext = readCsv(extPath, classAct);
        externalText.add(remove(ext, idx));
      }

    } else {
      train = new Instances(trainTotal);
      test = readCsv(testPath, classAct);
      for (String extPath : externalTestPath) {
        externalText.add(readCsv(extPath, classAct));
      }
    }

    return getEvaluation(train, test, externalText);
  }

  protected abstract double getEvaluation(
      Instances train, Instances internalTest, List<Instances> externalTests)
      throws ModelingException;

  protected Instances readCsv(String path, int act) throws ModelingException {
    Instances data = CSVManage.loadCSV(path);
    data.setClassIndex(act);
    return data;
  }

  protected void setClassIndex(Instances data, String nameAct) {
    if (data != null) {
      for (int i = 0; i < data.numAttributes(); i++) {
        if (data.attribute(i).name().equals(nameAct)) data.setClassIndex(i);
      }
    }
  }

  protected Set<String> getMDNames(Instances data) {
    Set<String> mdNames = new LinkedHashSet<>();
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != data.classIndex()) {
        mdNames.add(data.attribute(i).name());
      }
    }
    return mdNames;
  }

  protected Instances remove(Instances data, int[] pos) throws ModelingException {
    Remove rem = new Remove();
    rem.setAttributeIndicesArray(pos);
    rem.setInvertSelection(true);
    try {
      rem.setInputFormat(data);
      return Filter.useFilter(data, rem);
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.LOAD_DATASET_EXCEPTION.get(
          String.format("Problems loading dataset:%s", data.relationName()), ex);
    }
  }

  protected String getClassifierName(AbstractClassifier clasTmp) {
    String clasName = clasTmp.getClass().getSimpleName();
    if (clasName.equals("IBk")) {
      clasName = String.format("KNN(%s)", clasTmp.toString().split(" ")[3]);
    }
    return clasName;
  }
}
