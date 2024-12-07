# utils/evaluation.py

def evaluate(predictions, labels):
    """
    评估函数，计算预测值与真实值之间的准确率
    :param predictions: 预测结果（例如，分类标签）
    :param labels: 真实标签
    :return: 准确率
    """
    assert len(predictions) == len(labels), "预测结果和真实标签的长度不匹配"
    
    correct = sum([1 for p, l in zip(predictions, labels) if p == l])
    accuracy = correct / len(predictions)
    
    return accuracy
