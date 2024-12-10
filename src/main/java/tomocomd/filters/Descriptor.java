/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package tomocomd.filters;

/** @author Cesar */
public class Descriptor implements Comparable<Descriptor> {
  private final int position;
  private double min;
  private double max;
  private double entropy;
  private Bin[] bins;
  private final boolean isClass;

  public Descriptor(int position) {
    this(position, false);
  }

  public Descriptor(int position, boolean isClass) {
    this.position = position;
    this.isClass = isClass;
    this.entropy = 0d;
    this.bins = null;
  }

  @Override
  public final int compareTo(Descriptor desc) {
    if (entropy > desc.entropy) {
      return -1;
    } else if (entropy < desc.entropy) {
      return 1;
    }

    return 0;
  }

  public boolean isIsClass() {
    return isClass;
  }

  public double getMin() {
    return min;
  }

  public void setMin(double min) {
    this.min = min;
  }

  public double getMax() {
    return max;
  }

  public void setMax(double max) {
    this.max = max;
  }

  public int getPosition() {
    return position;
  }

  public double getEntropy() {
    return entropy;
  }

  public void setEntropy(double entropy) {
    this.entropy = entropy;
  }

  public Bin[] getBinsPartition() {
    return bins;
  }

  public void setBinsPartition(Bin[] bins) {
    this.bins = bins;
  }
}
