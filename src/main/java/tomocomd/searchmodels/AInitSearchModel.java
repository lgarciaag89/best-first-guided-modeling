package tomocomd.searchmodels;

import java.io.File;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;
import tomocomd.CSVManage;
import tomocomd.ModelingException;
import weka.attributeSelection.ASSearch;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;

public abstract class AInitSearchModel {
  protected String csvFile;
  protected String tuneSdf;
  protected String folderExtSdf;
  protected String act;
  protected List<AbstractClassifier> classifierList;
  protected List<ASSearch> searchList;

  protected Instances trainInstances;
  protected Instances tuneInstances;

  protected AInitSearchModel(
      String csvFile,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList) {
    this.csvFile = csvFile;
    this.tuneSdf = tuneSdf;
    this.folderExtSdf = folderExtSdf;
    this.act = act;
    this.classifierList = new LinkedList<>(classifierList);
    this.searchList = new LinkedList<>(searchList);
  }

  protected AInitSearchModel(
      Instances trainInstances,
      String csvFile,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList) {
    this.trainInstances = new Instances(trainInstances);
    this.csvFile = csvFile;
    this.tuneSdf = tuneSdf;
    this.folderExtSdf = folderExtSdf;
    this.act = act;
    this.classifierList = new LinkedList<>(classifierList);
    this.searchList = new LinkedList<>(searchList);
  }

  protected AInitSearchModel(
      Instances trainInstances,
      String csvFile,
      Instances tuneInstances,
      String tuneSdf,
      String folderExtSdf,
      String act,
      List<AbstractClassifier> classifierList,
      List<ASSearch> searchList) {
    this.trainInstances = new Instances(trainInstances);
    this.tuneInstances = Objects.isNull(tuneInstances) ? null : new Instances(tuneInstances);
    this.csvFile = csvFile;
    this.tuneSdf = tuneSdf;
    this.folderExtSdf = folderExtSdf;
    this.act = act;
    this.classifierList = new LinkedList<>(classifierList);
    this.searchList = new LinkedList<>(searchList);
  }

  protected List<String> loadingExternalTestPath() throws ModelingException {
    List<String> extInsts = new LinkedList<>();
    if (folderExtSdf == null) return extInsts;
    if (folderExtSdf.isEmpty()) return extInsts;

    File[] filesExt = new File(folderExtSdf).listFiles();
    if (Objects.isNull(filesExt)) return extInsts;
    Arrays.sort(filesExt);
    for (File ext : filesExt) {
      Instances extInst;
      try {
        extInst = CSVManage.loadCSV(ext.getAbsolutePath());
        extInst.setClassIndex(trainInstances.classIndex());
      } catch (Exception e) {
        throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
            String.format("Problems loading external dataset:%s", csvFile), e);
      }
      extInst.setClassIndex(trainInstances.classIndex());
      extInsts.add(ext.getAbsolutePath());
    }
    return extInsts;
  }

  public abstract void startSearchModel() throws ModelingException;
}
