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
public class RegressionModelInfo {

  private AbstractClassifier clas;
  private String[] desc;
  private String selName;
  private String claName;
  private int size;
  private long modelId;
  private double r2Loo;
  private double maeLoo;
  private double rmseLoo;
  private double r2Ext;
  private double maeExt;
  private double rmseExt;

  public RegressionModelInfo() {}

  private static double roundAvoid(double value) {
    int places = 6;
    double scale = Math.pow(10, places);
    return Math.round(value * scale) / scale;
  }

  public RegressionModelInfo(
      AbstractClassifier c,
      long modelId,
      int size,
      double r2Loo,
      double maeLoo,
      double rmseLoo,
      double r2Ext,
      double maeExt,
      double rmseExt,
      String[] descN)
      throws ModelingException {

    try {
      clas = (AbstractClassifier) AbstractClassifier.makeCopy(c);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(
          "Error coping the classifier", e);
    }

    this.modelId = modelId;
    this.r2Loo = roundAvoid(r2Loo);
    this.maeLoo = roundAvoid(maeLoo);
    this.r2Ext = roundAvoid(r2Ext);
    this.maeExt = roundAvoid(maeExt);
    this.rmseLoo = roundAvoid(rmseLoo);
    this.rmseExt = roundAvoid(rmseExt);
    this.size = size;
    setDesc(descN);
    selName = "";
    claName = "";
  }

  public RegressionModelInfo(
      AbstractClassifier c,
      long modelId,
      int size,
      double r2Loo,
      double maeLoo,
      double rmseLoo,
      double r2Ext,
      double maeExt,
      double rmseExt,
      String[] descN,
      String s,
      String cN)
      throws ModelingException {
    try {
      clas = (AbstractClassifier) AbstractClassifier.makeCopy(c);
    } catch (Exception e) {
      throw ModelingException.ExceptionType.CLASSIFIER_LOAD_EXCEPTION.get(
          "Error coping the classifier", e);
    }
    this.modelId = modelId;
    this.r2Loo = roundAvoid(r2Loo);
    this.maeLoo = roundAvoid(maeLoo);
    this.r2Ext = roundAvoid(r2Ext);
    this.maeExt = roundAvoid(maeExt);
    this.size = size;
    this.rmseLoo = roundAvoid(rmseLoo);
    this.rmseExt = roundAvoid(rmseExt);
    setDesc(descN);
    selName = s;
    claName = cN;
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

  public RegressionModelInfo(
      int modelId,
      int size,
      double r2Loo,
      double maeLoo,
      double rmseLoo,
      double r2Ext,
      double maeExt,
      double rmseExt) {
    this.modelId = modelId;
    this.r2Loo = roundAvoid(r2Loo);
    this.maeLoo = roundAvoid(maeLoo);
    this.r2Ext = roundAvoid(r2Ext);
    this.maeExt = roundAvoid(maeExt);
    this.rmseLoo = roundAvoid(rmseLoo);
    this.rmseExt = roundAvoid(rmseExt);
    this.size = size;
  }

  public RegressionModelInfo(RegressionModelInfo info) throws Exception {
    this.modelId = info.getModelId();
    this.r2Loo = info.getR2Loo();
    this.maeLoo = info.getMaeLoo();
    this.r2Ext = info.getR2Ext();
    this.maeExt = info.getMaeExt();
    this.clas = (AbstractClassifier) AbstractClassifier.makeCopy(info.getClas());
    this.size = info.getSize();
    this.claName = info.getClaName();
    this.selName = info.getSelName();
    this.rmseLoo = info.getRmseLoo();
    this.rmseExt = info.getRmseExt();
    setDesc(info.getDesc());
  }

  public String[] getDesc() {
    return Arrays.copyOf(desc, desc.length);
  }

  public void setDesc(String[] desc) {
    this.desc = Arrays.copyOf(desc, desc.length);
  }

  public int getSize() {
    return size;
  }

  public void setSize(int size) {
    this.size = size;
  }

  public AbstractClassifier getClas() {
    return clas;
  }

  public long getModelId() {
    return modelId;
  }

  public double getR2Loo() {
    return r2Loo;
  }

  public double getMaeLoo() {
    return maeLoo;
  }

  public double getR2Ext() {
    return r2Ext;
  }

  public double getMaeExt() {
    return maeExt;
  }

  public double getRmseLoo() {
    return rmseLoo;
  }

  public double getRmseExt() {
    return rmseExt;
  }
}
