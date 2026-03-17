from sklearn.metrics import f1_score, accuracy_score

def classification(y_pred, y_true, labels=[0, 1]):
    acc = accuracy_score(y_true, y_pred)
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=labels)
    micro_f1 = round(micro_f1, 4)
    macro_f1 = round(macro_f1, 4)
    acc = round(acc, 4)
    f1_support = round(f1_per_class[0], 4)
    f1_refute = round(f1_per_class[1], 4)
    
    return micro_f1, macro_f1, acc, f1_support, f1_refute