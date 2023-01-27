
learning_rate = 0.001
decay_rate = 0.02

for epoch in range(100):
    print(learning_rate / (1 + epoch * decay_rate))