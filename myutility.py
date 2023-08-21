import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torchvision.models as m
from mypythorchtool import Earlystopping
import torch.nn.utils.prune as prune
import gc

classes = ("CONTAINERS", "PLASTIC_BAG", "PLASTIC_BOTTLE", "TIN_CAN", "UNKNOWN")

def save_metrics(accuracy, precision, recall, f1_score, confusion_matrix, filename, model_folder):
    np.savetxt(model_folder + "/" + filename + "_accuracy.csv", np.array([accuracy]), delimiter=",", fmt="%.4f")
    np.savetxt(model_folder + "/" + filename + "_precision.csv", precision, delimiter=",", fmt="%.4f")
    np.savetxt(model_folder + "/" + filename + "_recall.csv", recall, delimiter=",", fmt="%.4f")
    np.savetxt(model_folder + "/" + filename + "_f1_score.csv", f1_score, delimiter=",", fmt="%.4f")
    np.savetxt(model_folder + "/" + filename + "_confusion_matrix.csv", confusion_matrix, delimiter=",", fmt="%d")

def evaluate(y_true, y_pred, num_classes):
    # Calcola la matrice di confusione
    confusion_matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[int(true)][int(pred)] += 1

    # Calcola l'accuratezza
    accuracy =100* sum(confusion_matrix[i][i] for i in range(num_classes)) / len(y_true)

    # Calcola la precision, recall e f1-score per ogni classe
    precision = [0 for _ in range(num_classes)]
    recall = [0 for _ in range(num_classes)]
    f1_score = [0 for _ in range(num_classes)]
    for i in range(num_classes):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes) if j != i)
        fn = sum(confusion_matrix[i][j] for j in range(num_classes) if j != i)
        precision[i] =100* (tp / (tp + fp)) if tp + fp != 0 else 0
        recall[i] =100* (tp / (tp + fn)) if tp + fn != 0 else 0
        f1_score[i] =(2 * precision[i] * recall[i] / (precision[i] + recall[i])) if precision[i] + recall[i] != 0 else 0


    return accuracy, precision, recall, f1_score, confusion_matrix

def evaluate_test(y_true, y_pred, file, epoch, accratezza):
    tp = [0] * 5  # Number of true positives for each class
    fp = [0] * 5  # Number of false positives for each class
    tn = [0] * 5  # Number of true negatives for each class
    fn = [0] * 5  # Number of false negatives for each class

    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        for j in range(5):
            if true_label == j:
                if pred_label == j:
                    tp[j] += 1
                else:
                    fn[j] += 1
            else:
                if pred_label == j:
                    fp[j] += 1
                else:
                    tn[j] += 1

    precision = [0] * 5
    recall = [0] * 5
    accuracy = 0
    f1score = [0] * 5

    for j in range(5):
        if tp[j] + fp[j] > 0:
            precision[j] = tp[j] / (tp[j] + fp[j])
        if tp[j] + fn[j] > 0:
            recall[j] = tp[j] / (tp[j] + fn[j])
        accuracy += tp[j]
        if precision[j] + recall[j] > 0:
            f1score[j] = 2 * (precision[j] * recall[j]) / (precision[j] + recall[j])

    accuracy /= len(y_true)
    
    # Cancella i file se esistono già
    if os.path.exists(file + "/_accuracy.csv"):
        os.remove(file + "/_accuracy.csv")
    if os.path.exists(file + "/_precision.csv"):
        os.remove(file + "/_precision.csv")
    if os.path.exists(file + "/_recall.csv"):
        os.remove(file + "/_recall.csv")
    if os.path.exists(file + "/_f1_score.csv"):
        os.remove(file + "/_f1_score.csv")
    if os.path.exists(file + "/_epoch_k.csv"):
        os.remove(file + "/_epoch_k.csv")

    # Salva i dati in file CSV
    np.savetxt(file + "/_accuracy.csv", np.array([accuracy]), delimiter=",", fmt="%.4f")
    np.savetxt(file + "/_precision.csv", np.array(precision), delimiter=",", fmt="%.4f")
    np.savetxt(file + "/_recall.csv", np.array(recall), delimiter=",", fmt="%.4f")
    np.savetxt(file + "/_f1_score.csv", np.array(f1score), delimiter=",", fmt="%.4f")
    np.savetxt(file + "/_epoch_k.csv", np.array([[epoch, accratezza]]), delimiter=",", fmt="%d,%.4f")




def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow((np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8))
    plt.show()

def GPUtest():
    try:
        torch.cuda.init()
    except:
        print("error during cuda initialization\n")
        exit(1)

    if not torch.cuda.is_available():
        print("error, cuda not available")
        exit(2)
    else:
        print("Cuda available, proceeding... \n")

def train(num_epochs, train_dl, val_dl, optimizer, net, criterion, history,file,file_info):
    best_accuracy = 0.0
    early_stopping = Earlystopping(patience=10, verbose=True,path=file)
    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
        running_train_loss = 0.0
        running_accuracy = 0.0
        running_vall_loss = 0.0
        total = 0

        # Training Loop
        for data in train_dl:
            inputs, outputs = data[0].to("cuda:0"), data[1].to("cuda:0")
            # get the input and real species as outputs; data is a list of [inputs, outputs]
            optimizer.zero_grad()  # zero the parameter gradients
            predicted_outputs = net(inputs)  # predict output from the model
            train_loss = criterion(
                predicted_outputs, outputs
            )  # calculate loss for the predicted output
            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_dl)

        # Validation Loop
        y_pred = []
        y_true = []
        with torch.no_grad():
            net.eval()
            for data in val_dl:
                inputs, outputs = data[0].to("cuda:0"), data[1].to("cuda:0")
                predicted_outputs = net(inputs)
                val_loss = criterion(predicted_outputs, outputs)

                # The label with the highest value will be our prediction
                _, predicted = torch.max(predicted_outputs, 1)
                running_vall_loss += val_loss.item()
                total += outputs.size(0)
                running_accuracy += (predicted == outputs).sum().item()
                y_true += outputs.tolist()
                y_pred += predicted.tolist()
                # Calculate validation loss value
        val_loss_value = running_vall_loss / len(val_dl)

        # Calculate accuracy as the number of correct predictions in the validation batch divided by the total number
        # of predictions done.
        accuracy = 100 * running_accuracy / total

        # save information
        history["train_loss"].append(train_loss_value)
        history["val_loss"].append(val_loss_value)
        history["accuracy"].append(accuracy)

        # Print the statistics of the epoch
        print(
            "Completed training epochs",
            epoch,
            "Training Loss is: %.4f" % train_loss_value,
            "Validation Loss is: %.4f" % val_loss_value,
            "Accuracy is %d %%" % accuracy,
        )
        early_stopping(val_loss_value, net,file)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if early_stopping.counter==0:
            evaluate_test(y_true,y_pred,file_info,epoch,accuracy)



def test(test_dl, net,model_number,architecture):
    print("\ninizio test")
    running_accuracy = 0
    total = 0
    y_pred = []
    y_true = []
    i=0
    with torch.no_grad():
        for data in test_dl:
            inputs, outputs = data[0].to("cuda:0"), data[1].to("cuda:0")
            outputs = outputs.to(torch.float32)
            predicted_outputs = net(inputs)
            _, predicted = torch.max(predicted_outputs, 1)
            total += outputs.size(0)
            running_accuracy += (predicted == outputs).sum().item()
            y_true += outputs.tolist()
            y_pred += predicted.tolist()
            torch.cuda.empty_cache()
            print("next",i)
            i+=1
        accuracy, precision, recall, f1score, cm = evaluate(y_true, y_pred,5)
        plot_confusion_matrix(cm,classes,net,dest(model_number,architecture))
    filename=str(net.__class__.__name__)
    save_metrics(accuracy, precision, recall, f1score, cm, filename,dest(model_number,architecture))
    print("\nfine test")

def plot_accuracies(history, net,num):
    """Plot the history of accuracies"""
    accuracies = history["accuracy"]
    plt.plot(accuracies, "-x")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs. No. of epochs")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    ax = plt.gca()  # get current axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(30)) # Imposta il numero di elementi sull'asse x
    ax.yaxis.set_major_locator(plt.MaxNLocator(30)) # Imposta il numero di elementi sull'asse y
    plt.savefig(dest_train(num)+ "/accuracies.jpg")

def plot_losses(history, net,num):
    """Plot the losses in each epoch"""
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    plt.plot(train_losses, "-bx")
    plt.plot(val_losses, "-rx")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training", "Validation"])
    plt.title("Loss vs. No. of epochs")
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    ax = plt.gca()  # get current axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(30)) # Imposta il numero di elementi sull'asse x
    ax.yaxis.set_major_locator(plt.MaxNLocator(30)) # Imposta il numero di elementi sull'asse y
    plt.savefig(dest_train(num) + "/loss.jpg")

def plot_confusion_matrix(confusion_matrix,name_classes,net,path):
    # Creazione del grafico
    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

    # Aggiunta di etichette e titolo
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(name_classes, rotation=20, fontsize=8) # aggiornamento della riga
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Classe predetta')
    ax.set_ylabel('Classe reale')
    ax.set_title('Matrice di Confusione')

    # Aggiunta dei numeri nella matrice di confusione
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = ax.text(j, i, confusion_matrix[i][j],
                           ha="center", va="center", color="gray")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig(path+"/" + str(net.__class__.__name__) + "confusion_matrix.jpg")

def select_model(model_number):
    if model_number == 1:
        model = m.mobilenet_v2()
    elif model_number == 2:
        model = m.mobilenet_v3_small()
    elif model_number == 3:
        model = m.mobilenet_v3_large()
    elif model_number == 4:
        model = m.shufflenet_v2_x2_0()
    elif model_number == 5:
        model = m.squeezenet1_1()
    elif model_number == 6:
        model = m.efficientnet_b0()
    elif model_number == 7:
        model = m.efficientnet_b1()
    elif model_number == 8:
        model = m.resnet18()
    elif model_number == 9:
        model = m.resnet34()
    elif model_number == 10:
        model = m.resnet50()
    elif model_number == 11:
        model = m.vgg11()
    elif model_number == 12:
        model = m.vgg16()
    elif model_number == 13:
        model = m.vgg19()
    else:
        raise ValueError("Invalid model number.")
    return model

def file(model_number):
    if model_number == 1:
        path = "./modelli/mobilenetv2.pt"
    elif model_number == 2:
        path = "./modelli/mobilenetv3_small.pt"
    elif model_number == 3:
        path = "./modelli/mobilenetv3_large.pt"
    elif model_number == 4:
        path = "./modelli/shufflenetv2.pt"
    elif model_number == 5:
        path = "./modelli/squeezenet1_1.pt"
    elif model_number == 6:
        path = "./modelli/efficientnet_b0.pt"
    elif model_number == 7:
        path = "./modelli/efficientnet_b1.pt"
    elif model_number == 8:
        path = "./modelli/resnet18.pt"
    elif model_number == 9:
        path = "./modelli/resnet34.pt"
    elif model_number == 10:
        path = "./modelli/resnet50.pt"
    elif model_number == 11:
        path = "./modelli/vgg11.pt"
    elif model_number == 12:
        path = "./modelli/vgg16.pt"
    elif model_number == 13:
        path = "./modelli/vgg19.pt"
    else:
        raise ValueError("Invalid model number.")
    return path


def dest(model_number, architecture):
    if architecture == "RTX":
        p1 = "TestModel"
    else:
        p1 = "TestModel_jetson"
    if model_number == 1:
        path = "./Info_model/" + p1 + "/mobilenetv2"
    elif model_number == 2:
        path = "./Info_model/" + p1 + "/mobilenetv3_small"
    elif model_number == 3:
        path = "./Info_model/" + p1 + "/mobilenetv3_large"
    elif model_number == 4:
        path = "./Info_model/" + p1 + "/shufflenetv2"
    elif model_number == 5:
        path = "./Info_model/" + p1 + "/squeezenet1_1"
    elif model_number == 6:
        path = "./Info_model/" + p1 + "/efficientnet_b0"
    elif model_number == 7:
        path = "./Info_model/" + p1 + "/efficientnet_b1"
    elif model_number == 8:
        path = "./Info_model/" + p1 + "/resnet18"
    elif model_number == 9:
        path = "./Info_model/" + p1 + "/resnet34"
    elif model_number == 10:
        path = "./Info_model/" + p1 + "/resnet50"
    elif model_number == 11:
        path = "./Info_model/" + p1 + "/vgg11"
    elif model_number == 12:
        path = "./Info_model/" + p1 + "/vgg16"
    elif model_number == 13:
        path = "./Info_model/" + p1 + "/vgg19"
    else:
        raise ValueError("Invalid model number.")
    return path

def dest_train(model_number):
    if model_number == 1:
        path = "./info_train/mobilenetv2"
    elif model_number == 2:
        path = "./info_train/mobilenetv3_small"
    elif model_number == 3:
        path = "./info_train/mobilenetv3_large"
    elif model_number == 4:
        path = "./info_train/shufflenetv2"
    elif model_number == 5:
        path = "./info_train/squeezenet1_1"
    elif model_number == 6:
        path = "./info_train/efficientnet_b0"
    elif model_number == 7:
        path = "./info_train/efficientnet_b1"
    elif model_number == 8:
        path = "./info_train/resnet18"
    elif model_number == 9:
        path = "./info_train/resnet34"
    elif model_number == 10:
        path = "./info_train/resnet50"
    elif model_number == 11:
        path = "./info_train/vgg11"
    elif model_number == 12:
        path = "./info_train/vgg16"
    elif model_number == 13:
        path = "./info_train/vgg19"
    else:
        raise ValueError("Invalid model number.")
    return path


def apply_pruning(model, prune_percent, save_name):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=prune_percent)
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=prune_percent)

    # Rimuovi i nodi pruned
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    # Salva il modello pruned
    torch.save(model.state_dict(), save_name)
