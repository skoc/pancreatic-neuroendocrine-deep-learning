import pandas as pd
import numpy as np
from typing import (Dict, IO, List, Tuple)

def calculate_confusion_matrix(all_labels: np.ndarray,
                               all_predicts: np.ndarray, classes: List[str],
                               num_classes: int) -> None:
    """
    Prints the confusion matrix from the given data.

    Args:
        all_labels: The ground truth labels.
        all_predicts: The predicted labels.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
    """
    remap_classes = {x: classes[x] for x in range(num_classes)}

    # Set print options.
    # Sources:
    #   1. https://stackoverflow.com/questions/42735541/customized-float-formatting-in-a-pandas-dataframe
    #   2. https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
    #   3. https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    actual = pd.Series(pd.Categorical(
        pd.Series(all_labels).replace(remap_classes), categories=classes),
                       name="Actual")

    predicted = pd.Series(pd.Categorical(
        pd.Series(all_predicts).replace(remap_classes), categories=classes),
                          name="Predicted")

    cm = pd.crosstab(index=actual, columns=predicted, normalize="index", dropna=False)

    cm.style.hide_index()
    print(cm)