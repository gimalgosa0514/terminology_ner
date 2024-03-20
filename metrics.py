import torch
import torchmetrics
import plotly.figure_factory as ff
#csv를 읽어들일 때 리스트를 문자열로 읽어들이는 문제 해결
from ast import literal_eval
import pandas as pd
from seqeval.metrics import classification_report

def str_to_list(x):
  try:
    if type(x) == str:
      return literal_eval(x)
    elif type(x) == list:
      return x
  except:
    return None

df = pd.read_csv("test_result.csv",encoding="utf-8")
df = df[df["labels"] != "labels"]

df["labels"] = df["labels"].apply(lambda x: str_to_list(x))
df["preds"] = df["preds"].apply(lambda x: str_to_list(x))

labels = df["labels"].tolist()
preds = df["preds"].tolist()

label = []
pred = []


for i in range(len(labels)):
  if labels[i] == None:
    break
  else:
    for ref, predd in zip(labels[i],preds[i]):
      label.append(ref)
      pred.append(predd)

true = [label]
predicted = [pred]



f1 = torchmetrics.F1Score(task = "multiclass", num_classes = 31, average ="macro")
label_str_to_id = lambda s:{"B-AF":0,
                            "B-AM":1,
                            "B-CV":2,
                            "B-DT":3,
                            "B-EV":4,
                            "B-FD":5,
                            "B-LC":6,
                            "B-MT":7,
                            "B-OG":8,
                            "B-PS":9,
                            "B-PT":10,
                            "B-QT":11,
                            "B-TI":12,
                            "B-TM":13,
                            "B-TR":14,
                            "I-AF":15,
                            "I-AM":16,
                            "I-CV":17,
                            "I-DT":18,
                            "I-EV":19,
                            "I-FD":20,
                            "I-LC":21,
                            "I-MT":22,
                            "I-OG":23,
                            "I-PS":24,
                            "I-PT":25,
                            "I-QT":26,
                            "I-TI":27,
                            "I-TM":28,
                            "I-TR":29,
                            "O":30}[s]

preds = torch.tensor(list(map(label_str_to_id,pred)))
labels = torch.tensor(list(map(label_str_to_id,label)))

f1_score = f1(preds, labels)
acc = torch.mean((preds == labels).float())


result = f"f1_score : {f1_score}" + "\n" + f"acc : {acc}" + "\n" + '\n'
#결과 파일에 쓰기
result = result + "class" +classification_report(true,predicted)
f = open("result.txt","w")
f.write(result)
f.close()

confusion_func = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=31)
confusion_matrix = confusion_func(preds, labels)
label = ["B-AF",
        "B-AM",
        "B-CV",
        "B-DT",
        "B-EV",
        "B-FD",
        "B-LC",
        "B-MT",
        "B-OG",
        "B-PS",
        "B-PT",
        "B-QT",
        "B-TI",
        "B-TM",
        "B-TR",
        "I-AF",
        "I-AM",
        "I-CV",
        "I-DT",
        "I-EV",
        "I-FD",
        "I-LC",
        "I-MT",
        "I-OG",
        "I-PS",
        "I-PT",
        "I-QT",
        "I-TI",
        "I-TM",
        "I-TR",
        "O"]

fig = ff.create_annotated_heatmap(confusion_matrix.numpy(), label, label)
fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
fig.write_image("confusion_matrix.png", format="png", scale=2)

