package tomocomd.filters;

import java.util.stream.IntStream;
import tomocomd.utils.PearsonCorrelationBetweenAttributes;
import tomocomd.utils.TriangularMatrixAsVector;
import weka.core.Instances;

public class PearsonFiltering {
  public PearsonFiltering() {}

  public static int[] getCorrelated(Instances data, double threshold) {
    TriangularMatrixAsVector triangularMatrixAsVector =
        PearsonCorrelationBetweenAttributes.computeCorrelationMatrix(data);
    int size = data.numAttributes();
    int classIndex = data.classIndex();

    boolean[] alreadyCorrelated = new boolean[size];

    return IntStream.range(0, size)
        .filter(i -> i != classIndex)
        .flatMap(
            i ->
                IntStream.range(0, i)
                    .filter(
                        j ->
                            j != classIndex
                                && !alreadyCorrelated[j]
                                && triangularMatrixAsVector.getEntry(i, j) > threshold)
                    .map(
                        j -> {
                          int correlatedIndex =
                              data.attribute(i).weight() > data.attribute(j).weight() ? j : i;
                          alreadyCorrelated[correlatedIndex] = true;
                          return correlatedIndex;
                        }))
        .distinct()
        .sorted()
        .toArray();
  }
}
