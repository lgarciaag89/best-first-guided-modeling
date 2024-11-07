package tomocomd.searchmodels.v3;

import java.io.File;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import tomocomd.CSVManage;
import tomocomd.ModelingException;
import tomocomd.searchmodels.v3.utils.MetricType;
import tomocomd.searchmodels.v3.utils.SearchPath;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.classifiers.AbstractClassifier;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

public class InitSearchModel {

    private final File csvTrainFileName;
    private final Instances trainData;

    private final Instances testData;
    private final String csvTuneFileName;

    private final List<String> externalTestNames;
    private final List<Instances> externalTestData;

    private final SearchPath searchPath;
    private final List<MetricType> metricTypes;
    private final List<ASSearch> searchAlgorithms;
    private final List<AbstractClassifier> classifiers;

    public InitSearchModel(
            File csvFile,
            File tuneCsv,
            File folderExtCsvs,
            String act,
            List<AbstractClassifier> classifierList,
            List<ASSearch> searchList,
            List<MetricType> metricTypes,
            SearchPath searchPath)
            throws ModelingException {

        this.searchPath = searchPath;
        this.metricTypes = new LinkedList<>(metricTypes);
        this.searchAlgorithms = new LinkedList<>(searchList);
        this.classifiers = new LinkedList<>(classifierList);
        this.csvTrainFileName = csvFile;

        // get train data
        Instances tempTrainData = CSVManage.loadCSV(csvFile.getAbsolutePath());
        tempTrainData.setRelationName(csvFile.getName());
        if (!setClassIndex(tempTrainData, act))
            throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
                    String.format("Problems loading %s target in train dataset", act));
        trainData = metricTypes.get(0).getProblemType().equals(MetricType.ProblemType.REGRESSION) ? tempTrainData :
                setTargetAttributeAsNominal(tempTrainData, tempTrainData.classIndex());

        // get tune data
        this.csvTuneFileName = Objects.nonNull(tuneCsv) ? tuneCsv.getName() : null;
        Instances tuneData =
                Objects.nonNull(tuneCsv) ? CSVManage.loadCSV(tuneCsv.getAbsolutePath()) : null;
        if (Objects.nonNull(tuneData)) {
            tuneData.setClassIndex(tempTrainData.classIndex());
            tuneData.setRelationName(tuneCsv.getName());
            testData = metricTypes.get(0).getProblemType().equals(MetricType.ProblemType.REGRESSION) ? tuneData :
                    setTargetAttributeAsNominal(tuneData, tempTrainData.classIndex());
        } else testData = null;

        externalTestData = loadingExternalTestPath(folderExtCsvs, tempTrainData.classIndex(),metricTypes.get(0).getProblemType());
        externalTestNames =
                externalTestData.stream().map(Instances::relationName).collect(Collectors.toList());
    }

    public void initSearchModel() {

        ExecutorService executorService = Executors.newWorkStealingPool();
        List<Runnable> tasks = new ArrayList<>();

        List<List<AbstractClassifier>> listClassifiers =
                searchPath.equals(SearchPath.SHORT)
                        ? Collections.singletonList(classifiers)
                        : classifiers.stream().map(Collections::singletonList).collect(Collectors.toList());

        // generating the list of tasks
        for (ASSearch search : searchAlgorithms) {
            for (MetricType metricType : metricTypes) {
                for (List<AbstractClassifier> classifierSubList : listClassifiers) {
                    String pathToSave =
                            String.format(
                                    "%s_models%s_%s_%s.csv",
                                    csvTrainFileName.getAbsolutePath(),
                                    classifierSubList.size() == 1
                                            ? "_" + classifierSubList.get(0).getClass().getSimpleName()
                                            : "",
                                    metricType.toString(),
                                    search.getClass().getSimpleName());
                    SearchModelEvaluator searchModelEvaluator =
                            new SearchModelEvaluator(
                                    csvTrainFileName.getName(),
                                    new Instances(trainData),
                                    csvTuneFileName,
                                    new Instances(testData),
                                    externalTestNames,
                                    copyExternalData(externalTestData),
                                    pathToSave,
                                    trainData.classIndex(),
                                    metricType,
                                    classifierSubList);

                    tasks.add(() -> startSearch(searchModelEvaluator, search));
                }
            }
        }

        List<Future<?>> futures = new ArrayList<>();
        try {
            for (Runnable task : tasks) {
                Future<?> future = executorService.submit(task);
                futures.add(future);
            }

            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        } finally {
            executorService.shutdown();
        }
    }

    private List<Instances> copyExternalData(List<Instances> externalTestData) {
        if (Objects.isNull(externalTestData)) return Collections.emptyList();
        return externalTestData.stream()
                .map(
                        instances -> {
                            Instances copy = new Instances(instances);
                            copy.setRelationName(instances.relationName());
                            return copy;
                        })
                .collect(Collectors.toList());
    }

    private void startSearch(SearchModelEvaluator searchModelEvaluator, ASSearch search) {
        AttributeSelection asSubset = new AttributeSelection();
        asSubset.setEvaluator(searchModelEvaluator);
        asSubset.setSearch(search);
        asSubset.setXval(false);

        try {
            Instances startTrain = new Instances(trainData);
            asSubset.SelectAttributes(startTrain);
            asSubset.selectedAttributes();
        } catch (Exception ex) {
            throw ModelingException.ExceptionType.BUILDING_MODEL_EXCEPTION.get(
                    "Problems building classification models", ex);
        }
    }

    //
    //    private void setclass(){
    //        if(metricType.getProblemType().equals(MetricType.ProblemType.CLASSIFICATION)) {
    //            trainData = setTargetAttributeAsNominal(tempTrainData, act);
    //            testData = setTargetAttributeAsNominal(tuneData, act);
    //            para los external tmb
    //        }else {
    //            trainData = tempTrainData;
    //        }
    //    }

    private List<Instances> loadingExternalTestPath(File folderExt, int classIdx, MetricType.ProblemType problemType)
            throws ModelingException {

        if (folderExt == null) return Collections.emptyList();

        if (!folderExt.exists() || !folderExt.isDirectory())
            throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
                    String.format("External folder do not  exist:%s", folderExt.getAbsolutePath()));

        File[] csvExts = folderExt.listFiles((dir, name) -> name.endsWith(".csv"));
        if (Objects.isNull(csvExts)) return Collections.emptyList();
        Arrays.sort(csvExts);

        List<Instances> extInsts = new ArrayList<>();
        for (File ext : csvExts) {
            Instances extInst;
            try {
                extInst = CSVManage.loadCSV(ext.getAbsolutePath());
                extInst.setClassIndex(classIdx);
                extInst.setRelationName(ext.getName());
                extInsts.add(problemType.equals(MetricType.ProblemType.CLASSIFICATION) ?
                        setTargetAttributeAsNominal(extInst, classIdx) : extInst);
            } catch (Exception e) {
                throw ModelingException.ExceptionType.CSV_FILE_WRITING_EXCEPTION.get(
                        String.format("Problems loading external dataset:%s", ext.getName()), e);
            }
        }
        return extInsts;
    }

    private Instances setTargetAttributeAsNominal(Instances data, int actIdx) {
        try {
            NumericToNominal filter = new NumericToNominal();
            filter.setAttributeIndicesArray(new int[]{actIdx});
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

    private static boolean setClassIndex(Instances data, String nameAct) {
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
}
