# best-first-guided-modeling

Using a best-first search algorithm to guide the modeling process.

![Java 8](https://img.shields.io/badge/Java-8-blue.svg)
![Maven](https://img.shields.io/badge/Maven-3.8.8-blue.svg)
[![Build And Release](https://github.com/cicese-biocom/best-first-guided-modeling/actions/workflows/maven_release.yml/badge.svg)](https://github.com/cicese-biocom/best-first-guided-modeling/actions/workflows/maven_release.yml)


![Latest Release](https://img.shields.io/github/v/release/cicese-biocom/best-first-guided-modeling?label=latest&style=flat-square)

## Description

<div style="text-align: justify;">
A java application to guide the modeling process using a best-first search algorithm.
Characteristics of the application:

1. Is based on Weka API
2. Is designed to be used with datasets in CSV format as input data.
3. Allows remove highly correlated attributes using Pearson correlation.
4. Allows remove attributes with a low Shannon entropy.
5. Allows to reduce the number of attributes usinng a consensus of:
    - Pearson correlation
    - Chi squared
    - Information gain
    - ReliefF
    - Symmetrical uncertainty
6. Performance on train dataset is obtained throw 10-CV.
7. Uses multiple evaluation metrics to assess the models.
8. The subsets for train the models are generates throw the best first search strategy.
9. Allows restart the search process from a given point.
10. The restore file(.status) is update automatically.


</div>

### Execution

The application can be executed using the following command:

```
usage: cmd [-c] [-e <arg>] [-f] [-h] [-m <arg>] [-o] [-p <arg>] [-pt
       <arg>] [-r] [-s] [-se <arg>] [-t <arg>] [-v] [-x <arg>]
 -c,--classification             Specifies that the problem is a
                                 classification problem. If it's a
                                 regression problem, do not set this
                                 option.
 -e,--endpoint <arg>             Target property, specifies the name of
                                 the variable to be used as the target.
 -f,--filter                     Execute filter operations (e.g., Shannon
                                 entropy (-se), Pearson correlation (-r)).
 -h,--help                       Displays this help message and exits.
 -m,--models <arg>               Space separate list of desired
                                 strategies. The strategies are:
                                 [KNN(C,R), RandomForest(C,R),
                                 Adaboost(C), AdditiveRegression(R),
                                 BayesNet(C), LogitBoost(C),
                                 RandomCommittee(C,R),
                                 SMO-PolyKernel(C,R), SMO-Puk(C,R),
                                 LinerRegression(R), 
                                 Bagging-SMO(C,R),  Bagging-KNN(C,R)],
                                 where C=Classification, R=Regression.
                                 Use "all" to apply all possible models
 -o,--reorder                    Reverses the order of the attributes.
 -p,--test <arg>                 input, test dataset, csv format
 -pt,--pearson-threshold <arg>   Pearson correlation threshold for
                                 eliminating highly correlated attributes.
 -r,--reduce                     Reduces the number of attributes.
  -re,--restart <arg>            Restart an incomplete execution, receive
                                 a file with the status of the incomplete
                                 execution.
 -s,--short                      If set, the search will be faster but may
                                 fall into local optima. Only one search
                                 will execute, and all classification
                                 algorithms will execute along the same
                                 path.
 -se,--se-threshold <arg>        Shannon entropy threshold for reducing
                                 the number of attributes.
 -t,--train <arg>                Input training dataset in CSV format.
 -v,--version                    Displays the version of the program and
                                 exits.
 -x,--external <arg>             External folder with several additional
                                 datasets in CSV format.
```

#### Examples

Applying a reducing strategy to the dataset:

```
java -jar best-first-guided-modeling-{version}.jar -r -e TARGET -t PATH_CSV
```

Applying a filtering and reduction strategy to the dataset: attributes with a Shannon Entropy (SE) lower than 0.25 will
be removed. Additionally, a Pearson correlation filter is applied. For each pair of attributes with a correlation higher
than 0.9, the attribute with the lower SE will be removed.

```
java -jar best-first-guided-modeling-{version}.jar -f -se 0.25 -pt 0.90 -r -e TARGET -t PATH_CSV
```

Applying a classification strategy, using Random Forest, KNN and SMO, to the dataset after filtering and reducing the
attributes:

```
java -jar best-first-guided-modeling-{version}.jar -f -se 0.25 -pt 0.90 -r -e TARGET -c -m RandomForest SMO KNN -t TRAIN_PATH_FILE_CSV -p TEST_PATH_FILE_CSV -x EXTERNAL_PATH_FOLDER
```

For restarting a previous execution, use the `-re` option with the path to the status file

```
java -jar best-first-guided-modeling-{version}.jar -re PATH_TO_STATUS_FILE
```

### Modeling strategies

The application uses the following strategies to guide the modeling process:
<table style="margin-left:auto; margin-right:auto;text-align: center;">
  <tr>
    <th>Strategy</th>
    <th>Is for classification</th>
    <th>Is for regression</th>
  </tr>
  <tr>
    <td>KNN</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
   <tr>
    <td>Random Forest</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>AdaBoost</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;"></td>
  </tr>
   <tr>
    <td>Additive Regression</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>Bayes Net</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;"></td>
  </tr>
<tr>
    <td>Logit Boost</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;"></td>
  </tr>
<tr>
    <td>Random Committee</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>SMO with Poly Kernel</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>SMO with Puk Kernel</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>Linear Regression</td>
    <td style="text-align: center;"></td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>Bagging with SMO</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
<tr>
    <td>Bagging with KNN</td>
    <td style="text-align: center;">X</td>
    <td style="text-align: center;">X</td>
  </tr>
</table>


# Format, Build and install

## Format
Use the following command to format the code:

```shell
mvn fmt:format
``` 

## Build and install
Use the following command to build and install the application:

```shell
mvn clean install
```

# Contributors

<a href="https://github.com/lgarciaag89">
    <img src="https://github.com/lgarciaag89.png" width="50" style="border-radius: 50%;" alt="Luis" />
</a>
<a href="https://github.com/cicese-biocom">
    <img src="https://github.com/cicese-biocom.png" width="50" style="border-radius: 50%;" alt="Cesar" />
</a>
