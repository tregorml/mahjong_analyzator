from mahjong.shanten import Shanten
import random
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from mahjong.tile import TilesConverter

def tile_name(index):
    if index < 9:
        return f'{index + 1} Man'
    elif index < 18:
        return f'{index - 8} Pin'
    elif index < 27:
        return f'{index - 17} Sou'
    else:
        honors = ['Ton', 'Nan', 'Sha', 'Pei', 'Haku', 'Hatsu', 'Chun']
        return honors[index - 27]

def analyze_hand(man='', pin='', sou='', honors=''):
    best_tiles = []
    best_shanten = 99
    shanten_calculator = Shanten()
    tiles_34 = TilesConverter.string_to_34_array(
        man=man, pin=pin, sou=sou, honors=honors
    )
    for i in range(34):
        if tiles_34[i] > 0:
            tiles_34[i] -= 1
            result = shanten_calculator.calculate_shanten(tiles_34)
            if result < best_shanten:
                best_shanten = result
                best_tiles = [i]
            elif result == best_shanten:
                best_tiles.append(i)
            tiles_34[i] += 1
    return {
        'best_shanten': best_shanten,
        'best_discards': [tile_name(i) for i in best_tiles]
    }

def generate_random_hand():
    deck = list(range(34)) * 4
    hand = random.sample(deck,13)
    return hand

def hand_to_34(hand):
    tiles_34 = [0] * 34
    for tile in hand:
        tiles_34[tile] += 1
    return tiles_34

def generate_training_sample():
    shanten_calculator = Shanten()
    deck = list(range(34)) * 4
    hand = random.sample(deck, 14)

    tiles_34 = hand_to_34(hand)

    best_shanten = 99
    best_discard = -1

    for i in range(34):
        if tiles_34[i] > 0:
            tiles_34[i] -= 1
            result = shanten_calculator.calculate_shanten(tiles_34)
            if result < best_shanten:
                best_shanten = result
                best_discard = i
            tiles_34[i] += 1

    return tiles_34, best_discard

def generate_dataset(n_samples=75000):
    X = []
    y = []

    for i in range(n_samples):
        tiles_34, best_discard = generate_training_sample()
        X.append(tiles_34)
        y.append(best_discard)

    return np.array(X), np.array(y)

class MahjongNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 34)
        )

    def forward(self, x):
        return self.net(x)

def predict_discard(man='', pin='', sou='', honors=''):
    tiles_34 = TilesConverter.string_to_34_array(
        man=man, pin=pin, sou=sou, honors=honors
    )
    x = torch.FloatTensor(tiles_34).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(x)
        predicted = output.argmax(dim=1).item()
    return tile_name(predicted)

if __name__ == '__main__':

    model = MahjongNet()
    X, y = generate_dataset(100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)


    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    y_train_t = torch.LongTensor(y_train)

    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(105):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test_t)
                predicted = test_output.argmax(dim=1)
                accuracy = (predicted == y_test_t).float().mean()
                print(f'epoch {epoch}: loss= {loss.item():.3f},acc = {accuracy:.3f}')
    torch.save(model.state_dict(), 'mahjong_model.pth')

result = analyze_hand(man='123456789',pin='111',sou='',honors='77')
print('algorithm: ',result)
prediction = predict_discard(man='123456789',pin='111',sou='',honors='77')
print('best drop: ',prediction)