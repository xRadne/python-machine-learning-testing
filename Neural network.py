import numpy as np


# Sigmoid function
def nonlin(x, deriv=False):
    if(deriv == True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))


# Input data
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target data
y = np.array([[0],
              [1],
              [1],
              [0]])

# Seed (for early debugging)
# np.random.seed(1)

# Weights/Synapses
syn0 = 2*np.random.random((2, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

# Training
print('Training:')
for j in range(0, 60000):

    # Layers
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # Backpropagation
    l2_error = y - l2
    if (j % 10000) == 0:
        print('Error: ' + str(np.mean(np.abs(l2_error))))

    # Calculate deltas
    l2_delta = l2_error * nonlin(l2, deriv=True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1, deriv=True)

    # Update our synapses
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

# Test the model
print('\n', 'Output after training')
for i in range(0, 4):
    print(x[i][0], ' XOR ', x[i][1], ' => ', l2[i][0])

a = int(2 * np.random.rand(1))
print("\nTest the model: ")
num = input("Random number: " + str(a) + ' XOR ')

l0 = np.array([a, int(num)])
l1 = nonlin(np.dot(l0, syn0))
l2 = nonlin(np.dot(l1, syn1))

print('')
print('Guess: ' + str(l2[0]))
print('Round: ' + str(int(np.round(l2[0]))) + ' => ' +
      str('True' if np.round(l2[0]) == 1 else 'False'))
