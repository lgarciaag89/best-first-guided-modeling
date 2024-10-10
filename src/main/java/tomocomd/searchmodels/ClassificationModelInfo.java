/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tomocomd.searchmodels;

import java.util.Arrays;
import tomocomd.ModelingException;
import weka.classifiers.AbstractClassifier;

/** @author potter */
public class ClassificationModelInfo {

  private AbstractClassifier clas;
  private String[] desc;
  private String selName;
  private String claName;
  private int size;
  private long modelId;
  private double accLoo;
  private double seLoo;
  private double spLoo;
  private double mccLoo;
  private double accExt;
  private double seExt;
  private double spExt;
  private double mccExt;

  public ClassificationModelInfo() {}

  public ClassificationModelInfo(
      int size,
      int modelId,
      double accLoo,
      double seLoo,
      double spLoo,
      double mccLoo,
      double accExt,
      double seExt,
      double spExt,
      double mccExt) {
    this.size = size;
    this.modelId = modelId;
    this.accLoo = (accLoo);
    this.seLoo = (seLoo);
    this.spLoo = (spLoo);
    this.mccLoo = (mccLoo);
    this.accExt = (accExt);
    this.seExt = (seExt);
    this.spExt = (spExt);
    this.mccExt = (mccExt);
  }

  public ClassificationModelInfo(
      AbstractClassifier clas,
      String[] desc,
      int size,
      long modelId,
      double accLoo,
      double seLoo,
      double spLoo,
      double mccLoo,
      double accExt,
      double seExt,
      double spExt,
      double mccExt)
      throws ModelingException {
    try {
      clas = (AbstractClassifier) AbstractClassifier.makeCopy(clas);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(
          "Error coping the classifier", e);
    }
    setDesc(desc);
    this.size = size;
    this.modelId = modelId;
    this.accLoo = accLoo;
    this.seLoo = seLoo;
    this.spLoo = spLoo;
    this.mccLoo = mccLoo;
    this.accExt = accExt;
    this.seExt = seExt;
    this.spExt = spExt;
    this.mccExt = mccExt;
  }

  public ClassificationModelInfo(
      AbstractClassifier clas,
      String[] desc,
      String selName,
      String claName,
      int size,
      long modelId,
      double accLoo,
      double seLoo,
      double spLoo,
      double mccLoo,
      double accExt,
      double seExt,
      double spExt,
      double mccExt)
      throws ModelingException {
    try {
      clas = (AbstractClassifier) AbstractClassifier.makeCopy(clas);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(
          "Error coping the classifier", e);
    }
    setDesc(desc);
    this.selName = selName;
    this.claName = claName;
    this.size = size;
    this.modelId = modelId;
    this.accLoo = accLoo;
    this.seLoo = seLoo;
    this.spLoo = spLoo;
    this.mccLoo = mccLoo;
    this.accExt = accExt;
    this.seExt = seExt;
    this.spExt = spExt;
    this.mccExt = mccExt;
  }

  public AbstractClassifier getClas() {
    return clas;
  }

  public void setClas(AbstractClassifier clas) {
    this.clas = clas;
  }

  public String getSelName() {
    return selName;
  }

  public void setSelName(String selName) {
    this.selName = selName;
  }

  public String getClaName() {
    return claName;
  }

  public void setClaName(String claName) {
    this.claName = claName;
  }

  public int getSize() {
    return size;
  }

  public void setSize(int size) {
    this.size = size;
  }

  public long getModelId() {
    return modelId;
  }

  public void setModelId(int modelId) {
    this.modelId = modelId;
  }

  public double getAccLoo() {
    return accLoo;
  }

  public void setAccLoo(double accLoo) {
    this.accLoo = accLoo;
  }

  public double getSeLoo() {
    return seLoo;
  }

  public void setSeLoo(double seLoo) {
    this.seLoo = seLoo;
  }

  public double getSpLoo() {
    return spLoo;
  }

  public void setSpLoo(double spLoo) {
    this.spLoo = spLoo;
  }

  public double getMccLoo() {
    return mccLoo;
  }

  public void setMccLoo(double mccLoo) {
    this.mccLoo = mccLoo;
  }

  public double getAccExt() {
    return accExt;
  }

  public void setAccExt(double accExt) {
    this.accExt = accExt;
  }

  public double getSeExt() {
    return seExt;
  }

  public void setSeExt(double seExt) {
    this.seExt = seExt;
  }

  public double getSpExt() {
    return spExt;
  }

  public void setSpExt(double spExt) {
    this.spExt = spExt;
  }

  public double getMccExt() {
    return mccExt;
  }

  public void setMccExt(double mccExt) {
    this.mccExt = mccExt;
  }

  public String[] getDesc() {
    return Arrays.copyOf(desc, desc.length);
  }

  public void setDesc(String[] desc) {
    this.desc = new String[desc.length];
    System.arraycopy(desc, 0, this.desc, 0, desc.length);
  }
}
