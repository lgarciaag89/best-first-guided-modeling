package tomocomd.utils;

import org.apache.commons.math3.stat.descriptive.moment.Kurtosis;
import tomocomd.filters.Bin;
import tomocomd.filters.Descriptor;

public class SEAttribute {

  public static double computeEntropy(double[] data) {
    Bin[] bins = new Bin[data.length];

    Descriptor descriptor = new Descriptor(0);
    if (fillBins(data, bins, descriptor)) {
      return calculateEntropy(bins);
    }
    return 0;
  }

  private static double calculateEntropy(Bin[] bins) {
    double entropy = 0.;
    double prob;
    for (Bin bin : bins) {
      if (bin == null) {
        return -1d;
      }

      if (bin.getCount() > 0) {
        prob = (double) bin.getCount() / bins.length;
        entropy -= prob * Math.log(prob);
      }
    }
    return entropy;
  }

  private static boolean fillBins(double[] values, Bin[] bins, Descriptor descriptor) {
    double min = Double.MAX_VALUE, max = Double.MIN_VALUE;
    for (double value : values) {
      if (Double.isNaN(value)) {
        return false;
      }
      min = Math.min(min, value);
      max = Math.max(max, value);
    }

    descriptor.setMin(min);
    descriptor.setMax(max);

    Kurtosis kurtosis = new Kurtosis();
    double kurtosis_val = kurtosis.evaluate(values);
    if (Double.isNaN(kurtosis_val)) {
      return false;
    }

    Bin bin;
    double binWidth, lower, upper, val;
    int binIndex;
    binWidth = (max - min) / bins.length;
    lower = min;

    for (int i = 0; i < bins.length; i++) {
      if (i == bins.length - 1) {
        bin = new Bin(lower, max);
      } else {
        upper = min + (i + 1) * binWidth;
        bin = new Bin(lower, upper);
        lower = upper;
      }
      bins[i] = bin;
    }

    for (double value : values) {
      binIndex = bins.length - 1;
      val = value;
      if (val < max) {
        double fraction = (val - min) / (max - min);
        if (fraction < 0.0) {
          fraction = 0.0;
        }
        binIndex = (int) (fraction * bins.length);
        // rounding could result in binIndex being equal to bins
        // which will cause an IndexOutOfBoundsException - see bug
        // report 1553088
        if (binIndex >= bins.length) {
          binIndex = bins.length - 1;
        }
      }
      bin = bins[binIndex];
      bin.incrementCount();
    }

    return true;
  }
}
