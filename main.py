import matplotlib.pylab
import matplotlib.pyplot
import torch
import torch.utils
import torch.utils.data
import math
import csv
import tqdm
import pickle
import matplotlib

class EconomicData(torch.utils.data.Dataset):
    def __init__(self):
        super(EconomicData, self).__init__()
        datas1 = []
        with open("leading_indicators.csv", newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            keys = reader.fieldnames
            for row in reader:
                row_info = []
                for key in keys:
                    assert key in row
                    if key not in ["time"] and key in ["United States"]:
                        row_info.append(float(row[key])-100)
                datas1.append(row_info)

        datas2 = []
        with open("unemployed.csv", newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            keys = reader.fieldnames

            for row in reader:
                row_info = []
                for key in keys:
                    assert key in row
                    if key not in ["time"] and key in ["United States"]:
                        row_info.append(float(row[key])/100)
                datas2.append(row_info)

        datas3=[]
        with open("CPI_PCE.csv", newline='', encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            keys = reader.fieldnames
            print(keys)
            for row in reader:
                row_info = []
                for key in keys:
                    assert key in row
                    if key not in ["time"]:
                        if key == "PCE" or key == "CPI":
                            row_info.append(float(row[key])/5)
                        elif key == "PMI":
                            row_info.append((float(row[key])-50)/10)
                        elif key == "Interest_Rate":
                            row_info.append((float(row[key]))/5)
                datas3.append(row_info)
        
        self.datas = []
        print(len(datas3), len(datas2),len(datas1))
        assert(len(datas1) == len(datas2))
        assert(len(datas3) == len(datas2))
        posi = 0
        negt = 0
        bal = 0
        for i in range(len(datas2)-1):
            if (datas3[i+1][-1] - datas3[i][-1])!=0:
                if (datas3[i+1][-1] - datas3[i][-1]) > 0:
                    for mul in range(6):
                        posi += 1
                        self.datas.append(
                            (datas1[i]+datas2[i]+datas3[i], datas3[i+1][-1])
                        )
                else:
                    for mul in range(30):
                        negt += 1
                        self.datas.append(
                            (datas1[i]+datas2[i]+datas3[i], datas3[i+1][-1])
                        )
            else:
                pass
                bal += 1
                self.datas.append(
                        (datas1[i]+datas2[i]+datas3[i], datas3[i+1][-1])
                    )

        print(posi, negt, bal)
        input()
        self.num_feature = len(datas1[-1]+datas2[-1]+datas3[-1])
        self.x_last = torch.Tensor(datas1[-1]+datas2[-1]+datas3[-1]).cuda().unsqueeze(dim=0)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return torch.Tensor(self.datas[index][0]), torch.Tensor([self.datas[index][1]])

def main():
    dataset = EconomicData()
    #set1, set2 = torch.utils.data.random_split(dataset, lengths=[0.99, 0.01])
    train_set = dataset
    val_set = dataset
    num_epoch = 4096
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)

    
    network = torch.nn.Sequential(
        torch.nn.Linear(in_features=dataset.num_feature, out_features=256, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features=256, out_features=128, bias=True),
        torch.nn.ReLU(inplace=True),

        torch.nn.Linear(in_features=128, out_features=32, bias=True),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(in_features=32, out_features=1, bias=True),
    ).cuda()

    loss_fn = torch.nn.MSELoss()
    optim = torch.optim.SGD(params=network.parameters(),lr=0.001, momentum=0.8, weight_decay=0.01)
    sch   = torch.optim.lr_scheduler.StepLR(optimizer=optim, step_size=1024, gamma=0.1, last_epoch=-1, verbose=True)
    list_loss = []
    for epoch_id in range(num_epoch):
        l_sum = 0
        for i,(x,y) in enumerate(tqdm.tqdm(train_dataloader, disable=True)):
            x = x.cuda()
            y = y.cuda()
            y_est = network(x)
            l = loss_fn(y,y_est)
            l_sum = l_sum + l
            optim.zero_grad()
            l.backward()
            optim.step()
        sch.step()

        l_sum_val = 0
        for i,(x,y) in enumerate(tqdm.tqdm(val_dataloader, disable=True)):
            x = x.cuda()
            y = y.cuda()
            y_est = network(x)
            l = loss_fn(y,y_est)
            l_sum_val = l_sum_val + l

        print((l_sum/len(train_dataloader)).item(),(l_sum_val/len(val_dataloader)).item())
        list_loss.append((l_sum/len(train_dataloader)).item())
        torch.save(network.state_dict(), "./trained_network")

    x = list(range(len(list_loss)))
    matplotlib.pyplot.plot(x, list_loss)
    matplotlib.pyplot.xlabel("Epoch")
    matplotlib.pyplot.ylabel("Loss")
    matplotlib.pyplot.title("Training Process")
    matplotlib.pyplot.show()

    pickle.dump(list_loss, "./loss.pkl")
    print(network(dataset.x_last)*5)

if __name__ == "__main__":
    main()