import torch
import torch.nn as nn

torch.manual_seed(0)

# A tiny RNN so the numbers fit on the page: 3-dimensional words, 4-dimensional state.
rnn = nn.RNN(input_size=3, hidden_size=4, batch_first=True)

words = torch.tensor([[[1.0, 0.0, 0.0],    # қосымша
                       [0.0, 1.0, 0.0],    # өте
                       [0.0, 0.0, 1.0],    # жақсы
                       [1.0, 1.0, 0.0]]])  # емес

# --- what nn.RNN does, written out ----------------------------------------
W_ih, W_hh = rnn.weight_ih_l0, rnn.weight_hh_l0
b_ih, b_hh = rnn.bias_ih_l0, rnn.bias_hh_l0

h = torch.zeros(4)                                   # the state starts at zero
print("h0 (before any word):", h.numpy().round(3))
for step in range(4):
    x = words[0, step]
    h = torch.tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)   # <- the whole recurrence
    print(f"h{step + 1} after word {step + 1}   :", h.detach().numpy().round(3))

# --- the same thing, done by PyTorch --------------------------------------
outputs, last = rnn(words)
print("\nPyTorch's last hidden state:", last[0, 0].detach().numpy().round(3))
print("matches our loop           :", bool(torch.allclose(h, last[0, 0], atol=1e-6)))
print("\noutputs holds every step   :", tuple(outputs.shape), "= (batch, words, hidden)")
print("the last row of outputs is the last hidden state:",
      bool(torch.allclose(outputs[0, -1], last[0, 0])))
