import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    """
    Returns a dictionary containing accuracy, precision, recall, and f1 rounded to six decimals.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Global Accuracy
    total_samples = len(y_true)
    if total_samples == 0:
        accuracy = 0.0
    else:
        accuracy = np.mean(y_true == y_pred)
        
    # Helper function for safe division
    def safe_div(numerator, denominator):
        return numerator / denominator if denominator > 0 else 0.0

    # Get unique classes across both arrays
    classes = np.unique(np.concatenate([y_true, y_pred]))

    if average == 'micro':
        # Aggregate counts globally across all classes
        tp_total = np.sum(y_true == y_pred)
        fp_total = np.sum(y_true != y_pred)
        fn_total = np.sum(y_true != y_pred)

        precision = safe_div(tp_total, tp_total + fp_total)
        recall = safe_div(tp_total, tp_total + fn_total)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == 'binary':
        # Compute metrics specifically for pos_label
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average in ('macro', 'weighted'):
        # Compute per-class TP, FP, FN, P_c, R_c, F1_c
        precisions = []
        recalls = []
        f1s = []
        supports = []

        for c in classes:
            tp_c = np.sum((y_true == c) & (y_pred == c))
            fp_c = np.sum((y_true != c) & (y_pred == c))
            fn_c = np.sum((y_true == c) & (y_pred != c))
            support_c = np.sum(y_true == c)

            p_c = safe_div(tp_c, tp_c + fp_c)
            r_c = safe_div(tp_c, tp_c + fn_c)
            f1_c = safe_div(2 * p_c * r_c, p_c + r_c)

            precisions.append(p_c)
            recalls.append(r_c)
            f1s.append(f1_c)
            supports.append(support_c)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        f1s = np.array(f1s)
        supports = np.array(supports)

        if average == 'macro':
            precision = np.mean(precisions) if len(precisions) > 0 else 0.0
            recall = np.mean(recalls) if len(recalls) > 0 else 0.0
            f1 = np.mean(f1s) if len(f1s) > 0 else 0.0

        elif average == 'weighted':
            total_support = np.sum(supports)
            if total_support > 0:
                precision = np.sum(precisions * supports) / total_support
                recall = np.sum(recalls * supports) / total_support
                f1 = np.sum(f1s * supports) / total_support
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0
    else:
        raise ValueError(f"Unsupported average method: {average}. Choose from 'micro', 'macro', 'weighted', or 'binary'.")

    return {
        'accuracy': round(float(accuracy), 6),
        'precision': round(float(precision), 6),
        'recall': round(float(recall), 6),
        'f1': round(float(f1), 6)
    }    
    pass