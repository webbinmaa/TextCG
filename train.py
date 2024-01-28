import os
import torch
import torch.nn as nn
from transformers import AdamW
from torch.utils.data import DataLoader
# from model1 import Classifier
# from model3 import Classifier_uncross
# from model3 import Classifier_unM
# from model3 import Classifier_fusion
from model3 import Classifier_unbilstm
from dataset import Dataset
from tqdm import tqdm
from sklearn import metrics



def main():

    batch_size = 10
    device = 'cpu'
    epochs = 20
    learning_rate =5e-6
    bert_path = './bert-base-uncased'


    # obtain dataset
    train_dataset = Dataset('data/sst1/train')
    test_dataset = Dataset('data/sst1/test')

    num_labels = len(train_dataset.labels)

    # Initialization model
    model = Classifier_unbilstm(bert_path, batch_size, num_labels).to(device)
    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # Loss function:cross entropy
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        losses = 0
        accuracy = 0

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # drop-last丢弃最后1个不满的batch
        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id, adj in train_bar:

            model.zero_grad()
            train_bar.set_description('Epoch %i train' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                adj=adj.to(device)

            )
            # loss
            kl_loss = criterion(output, label_id.to(device))

            loss = kl_loss
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(pred_labels)  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)

        print('\tTrain ACC:', average_acc, '\tLoss:', average_loss)

        model.eval()
        losses = 0
        acc_test = 0
        pred_labels = []
        true_labels = []
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        valid_bar = tqdm(test_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id, adj in valid_bar:
            valid_bar.set_description('Epoch %i valid' % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                adj=adj.to(device)
            )

            kl_loss_v = criterion(output, label_id.to(device))

            loss = kl_loss_v
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)
            acc = torch.sum((pred_label == label_id.to(device)).int()).item() / len(pred_label)
            acc_test += acc

            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss_test = losses / len(test_dataloader)
        average_acc_test = acc_test / len(test_dataloader)

        print('\tTest ACC:', average_acc_test, '\tLoss:', average_loss_test)

        report = metrics.classification_report(true_labels, pred_labels, labels=test_dataset.labels_id,
                                               target_names=test_dataset.labels)
        print('* Classification Report:')
        print(report)

        f1 = metrics.f1_score(true_labels, pred_labels, labels=test_dataset.labels_id, average='micro',zero_division=0)

        if not os.path.exists('models'):
            os.makedirs('models')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'models/sst1.pkl')


if __name__ == '__main__':
    main()