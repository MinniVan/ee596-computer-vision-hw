import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt

def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4
    # load train split
    trainset = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    # take one rando mini-bathc
    dataier = iter(trainloader)

    images, labels = next(dataier)
    
    return images, labels

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
def train_classifier():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    #print('Finished Training')
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)
    #print("Saved trained weights to: ", PATH)

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    # Loads dataset
    batch_size=4
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

def get_first_layer_weights():
    net = Net()
    # TODO: load the trained weights
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))  # load saved weights
    first_weight = net.conv1.weight.data.clone() # copy tensor
    return first_weight

def get_second_layer_weights():
    net = Net()
    # TODO: load the trained weights
    PATH = './cifar_net_2epoch.pth'
    net.load_state_dict(torch.load(PATH, weights_only=True))
    second_weight = net.conv2.weight.data.clone()# TODO: get conv2 weights (exclude bias)
    return second_weight

def hyperparameter_sweep():
    '''
    Reuse the CNN and training code from Question 2
    Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
    During training, record the training loss every 2000 iterations
    compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
    After training, plot three curves
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # train/test sets
    trainset_full = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=True,
        download=True,
        transform=transform
    )

    testset_full = torchvision.datasets.CIFAR10(
        root='./cifar10',
        train=False,
        download=True,
        transform=transform
    )

    def make_loader(dataset, shuffle_flag):
        batch_size = 4
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    
    # helr to sample k random images and labels from a dataset.
    # Returns:
    #        images: Tensor [k, 3, 32, 32]
    #        labels: LongTensor [k]
    def sample_subset(dataset, k=1000):
        idxs = random.sample(range(len(dataset)), k)
        imgs_list = []
        labs_list = []
        for idx in idxs:
            x, y = dataset[idx] 
            imgs_list.append(x.unsqueeze(0))  # [1,3,32,32]
            labs_list.append(y)
        imgs = torch.cat(imgs_list, dim=0)    # [k,3,32,32]
        labs = torch.tensor(labs_list).long() # [k]
        return imgs, labs
    
    # hlper to comupute classification error% on given tensors (images, labels)
    # error% = 100*(1-accuarcy)
    def eval_error_percent(net, imgs, labels):
        with torch.no_grad():
            outputs = net(imgs)                # [N,10]
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total = labels.size(0)
            err = 100.0 * (1.0 - (correct / total))
        return err
    # train new net for 2 pecos with a given learning rate: lr_value
    # Returns: iters_log, loss_log, train_err_log, test_err_log
    def train_once(lr_value):
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr_value, momentum=0.9)

        trainloader = make_loader(trainset_full, shuffle_flag=True)
         # fixed 1k-train subset and 1k-test subset for this run
        train_eval_imgs, train_eval_labs = sample_subset(trainset_full, k=1000)
        test_eval_imgs, test_eval_labs = sample_subset(testset_full, k=1000)

        iters_log = []
        loss_log = []
        train_err_log = []
        test_err_log = []

        running_loss = 0.0
        iter_count = 0

        for epoch in range(2):
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                iter_count += 1
                running_loss += loss.item()

                # record stats every 2000 iters
                if (iter_count % 2000) == 0:
                    avg_loss = running_loss / 2000.0
                    running_loss = 0.0

                    # training error on 1k train subset
                    train_err = eval_error_percent(net, train_eval_imgs, train_eval_labs)
                    # test error on 1k test subset
                    test_err = eval_error_percent(net, test_eval_imgs, test_eval_labs)

                    iters_log.append(iter_count)
                    loss_log.append(avg_loss)
                    train_err_log.append(train_err)
                    test_err_log.append(test_err)

                    #print(f"[lr={lr_value} epoch {epoch+1} iter {iter_count}] "
                    #        f"loss={avg_loss:.3f}, "
                    #        f"train_err={train_err:.2f}%, "
                    #        f"test_err={test_err:.2f}%")
                    
        return iters_log, loss_log, train_err_log, test_err_log
    
    # 3 trainings with differ learning rates
    learning_rates = [0.01, 0.001, 0.0001]
    sweep_results = {} # sweep_results[lr] = (iters_log, loss_log, train_err_log, test_err_log)

    for lr in learning_rates:
        sweep_results[lr] = train_once(lr)
    
    '''
    # plots
    # 1. training loss vs iteration
    plt.figure()
    for lr in learning_rates:
        iters_log, loss_log, _, _ = sweep_results[lr]
        plt.plot(iters_log, loss_log, label=f"lr={lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Training loss (avg over last 2000 iters)")
    plt.title("Training Loss vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2) training error vs iteration
    plt.figure()
    for lr in learning_rates:
        iters_log, _, train_err_log, _ = sweep_results[lr]
        plt.plot(iters_log, train_err_log, label=f"lr={lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Training error (%) on 1k-train subset")
    plt.title("Training Error vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

        # 3) test error vs iteration
    plt.figure()
    for lr in learning_rates:
        iters_log, _, _, test_err_log = sweep_results[lr]
        plt.plot(iters_log, test_err_log, label=f"lr={lr}")
        # lower test_err_log is better
    plt.xlabel("Iteration")
    plt.ylabel("Test error (%) on 1k-test subset")
    plt.title("Test Error vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.show()

    '''
    return None

if __name__ == "__main__":
    # your text code here

    # generate and show the visualization for the report
    #isualize_four_images()
    images, labels = CIFAR10_dataset_a()
    train_classifier()
    evalNetwork()
    hyperparameter_sweep()