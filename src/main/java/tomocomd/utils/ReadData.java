package tomocomd.utils;

import java.io.File;
import java.util.*;
import tomocomd.CSVManage;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class ReadData {

  public static Instances readData(File csvFile, String act, boolean isClassification)
      throws ModelingException {
    if (Objects.isNull(csvFile)) return null;
    Instances tempData = CSVManage.loadCSV(csvFile.getAbsolutePath());
    if (Objects.isNull(tempData))
      throw ModelingException.ExceptionType.CSV_FILE_LOADING_EXCEPTION.get(
          String.format("Problems loading %s dataset", csvFile.getName()));
    tempData.setRelationName(csvFile.getName());
    if (!setClassIndex(tempData, act))
      throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
          String.format("Problems loading %s target in train dataset", act));
    return isClassification
        ? setTargetAttributeAsNominal(tempData, tempData.classIndex())
        : tempData;
  }

  public static List<Instances> loadingExternalTestPath(
      File folderExt, String act, boolean isClassification) throws ModelingException {

    if (folderExt == null) return Collections.emptyList();

    if (!folderExt.exists() || !folderExt.isDirectory())
      throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
          String.format("External folder do not  exist:%s", folderExt.getAbsolutePath()));

    File[] csvExts = folderExt.listFiles((dir, name) -> name.endsWith(".csv"));
    if (Objects.isNull(csvExts)) return Collections.emptyList();
    Arrays.sort(csvExts);

    List<Instances> extInsts = new ArrayList<>();
    for (File ext : csvExts) {
      try {
        extInsts.add(readData(ext, act, isClassification));
      } catch (Exception e) {
        throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
            String.format("Problems loading external dataset:%s", ext.getName()), e);
      }
    }
    return extInsts;
  }

  public static boolean setClassIndex(Instances data, String nameAct) {
    if (data != null) {
      for (int i = 0; i < data.numAttributes(); i++) {
        if (data.attribute(i).name().equals(nameAct)) {
          data.setClassIndex(i);
          return true;
        }
      }
    }
    return false;
  }

  private static Instances setTargetAttributeAsNominal(Instances data, int actIdx) {
    try {
      NumericToNominal filter = new NumericToNominal();
      filter.setAttributeIndicesArray(new int[] {actIdx});
      filter.setInputFormat(data);
      Instances newData = Filter.useFilter(data, filter);
      newData.setClassIndex(actIdx);
      newData.setRelationName(data.relationName());
      return newData;
    } catch (Exception ex) {
      throw ModelingException.ExceptionType.CSV_FILE_LOADING_EXCEPTION.get(
          String.format(
              "Problems setting target attribute idx %d as nominal for %s dataset",
              actIdx, data.relationName()),
          ex);
    }
  }
}
