package tomocomd;

public class ModelingException extends RuntimeException {
  final ExceptionType type;

  public ModelingException(ExceptionType type) {
    super(type.getMessage());
    this.type = type;
  }

  public ModelingException(ExceptionType type, Throwable cause) {
    super(type.getMessage(), cause);
    this.type = type;
  }

  public ModelingException(ExceptionType type, Throwable cause, String message) {
    super(type.formatMessage(message), cause);
    this.type = type;
  }

  public ModelingException(ExceptionType type, String message) {
    super(type.formatMessage(message));
    this.type = type;
  }

  public enum ExceptionType {
    CSV_FILE_WRITING_EXCEPTION("Error writing csv file"),
    CSV_FILE_LOADING_EXCEPTION("Error loading csv file"),
    LOAD_DATASET_EXCEPTION("Error writing csv file"),
    BUILDING_MODEL_EXCEPTION("Error building model"),
    CLASSIFIER_LOAD_EXCEPTION("Error loading the classifier");

    private final String message;

    ExceptionType(String s) {
      this.message = s;
    }

    public String getMessage() {
      return message;
    }

    public String formatMessage(String message) {
      return String.format("%s: %s", this.message, message);
    }

    public ModelingException get() {
      return new ModelingException(this);
    }

    public ModelingException get(String message) {
      return new ModelingException(this, message);
    }

    public ModelingException get(Throwable cause) {
      return new ModelingException(this, cause);
    }

    public ModelingException get(String message, Throwable cause) {
      return new ModelingException(this, cause, message);
    }
  }
}
