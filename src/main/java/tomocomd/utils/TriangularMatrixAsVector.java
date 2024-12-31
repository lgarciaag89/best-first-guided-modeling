package tomocomd.utils;

public class TriangularMatrixAsVector {

  private final double[] vector;
  private final int size;

  public TriangularMatrixAsVector(int size) {
    if (size <= 0) {
      throw new IllegalArgumentException("Matrix size must be positive.");
    }
    this.size = size;
    this.vector = new double[(size * (size + 1)) / 2]; // Allocate vector storage
  }

  /** Computes the index in the vector for given row and column. */
  private int index(int row, int col) {
    if (row < col) {
      throw new IllegalArgumentException("Only lower triangular elements are allowed.");
    }
    if (row >= size || col >= size || row < 0 || col < 0) {
      throw new IndexOutOfBoundsException("Invalid row or column index.");
    }
    return (row * (row + 1)) / 2 + col;
  }

  /** Sets the value at the specified row and column. */
  public void setEntry(int row, int col, double value) {
    vector[index(row, col)] = value;
  }

  /** Gets the value at the specified row and column. */
  public double getEntry(int row, int col) {
    return row < col ? 0.0 : vector[index(row, col)];
  }

  /** Prints the matrix in a readable format. */
  public void printMatrix() {
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        System.out.print((i < j ? 0.0 : getEntry(i, j)) + " ");
      }
      System.out.println();
    }
  }

  public double[] getVectorWithOutDiagonal() {
    double[] vectorWithOutDiagonal = new double[vector.length - size];
    int count = 0;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < i; j++) {
        vectorWithOutDiagonal[count++] = getEntry(i, j);
      }
    }
    return vectorWithOutDiagonal;
  }

  public double[] getVector() {
    return vector;
  }
}
