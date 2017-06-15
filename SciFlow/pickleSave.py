import NNFlow
import pickle

silvia = NNFlow.MLPRegFlow(hidden_layer_sizes=(45,), max_iter=80)
pickle.dump(silvia, open('../tests/model.pickl','wb'))