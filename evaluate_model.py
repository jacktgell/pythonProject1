import numpy as np


def get_dataset_balance(arr):
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    print("Unique rows: ", unique)
    print("Frequencies: ", counts)

def test_model_on_target(x, y, model, target_val):
    featidx = np.where((y == target_val).all(axis=1))
    sub_x = [np.array(x[idx]) for idx in featidx][0]
    sub_y = [np.array(y[idx]) for idx in featidx][0]
    return model.evaluate(sub_x, sub_y, return_dict=True)
def test_model_on_target_lin(x, y, model):
    predictions = model.predict(x)
    accuracy = {'buy':[], 'sell':[], 'hold':[]}

    try:
        for result in zip(predictions, y):
            if result[1] > 1.001:
                if result[0] > 1.001:
                    accuracy['buy'].append(1)
                else:
                    accuracy['buy'].append(0)
            elif result[1] < 0.999:
                if result[0] < 0.999:
                    accuracy['sell'].append(1)
                else:
                    accuracy['sell'].append(0)
            else:
                if result[0] > 0.999 and result[1] < 1.001:
                    accuracy['hold'].append(1)
                else:
                    accuracy['hold'].append(0)
    except:
        a = 1

    accuracy['buy'] = sum(accuracy['buy'])/len(accuracy['buy']) if len(accuracy['buy']) else 0
    accuracy['sell'] = sum(accuracy['sell'])/len(accuracy['sell']) if len(accuracy['sell']) else 0
    accuracy['hold'] = sum(accuracy['hold'])/len(accuracy['hold']) if len(accuracy['hold']) else 0
    return accuracy
