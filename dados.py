import numpy as np
def criando_as_entradas(text):
  """ Criando uma matriz one-hot
  """
  vocabulario = []
  for i in (list(train_data.keys()) + list(test_data.keys())):
      vocabulario += i.split()
  vocabulario = list(np.unique(vocabulario))
  vocab_size = len(vocabulario)

  word_to_idx = { w: i for i, w in enumerate(vocabulario) }
  idx_to_word = { i: w for i, w in enumerate(vocabulario) }

  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs,vocab_size,

train_data = {
  'bom': True,
  'ruim': False,
  'feliz': True,
  'triste': False,
  'não bom': False,
  'não ruim': True,
  'não feliz': False,
  'não triste': True,
  'muito bom': True,
  'muito ruim': False,
  'muito feliz': True,
  'muito triste': False,
  'estou feliz': True,
  'isso é bom': True,
  'estou ruim': False,
  'isso é ruim': False,
  'estou triste': False,
  'isso é triste': False,
  'não estou feliz': False,
  'isso não é bom': False,
  'não estou ruim': True,
  'isso não é triste': True,
  'estou muito feliz': True,
  'isso é muito bom': True,
  'estou muito ruim': False,
  'isso é muito triste': False,
  'isso é muito feliz': True,
  'estou bem, não ruim': True,
  'isso é bom, não ruim': True,
  'estou ruim, não bom': False,
  'estou bom e feliz': True,
  'isso não é bom e não feliz': False,
  'não estou nada bom': False,
  'não estou nada ruim': True,
  'não estou nada feliz': False,
  'isso não é nada triste': True,
  'isso não é nada feliz': False,
  'estou bem agora': True,
  'estou ruim agora': False,
  'isso é ruim agora': False,
  'estou triste agora': False,
  'eu estava bem antes': True,
  'eu estava feliz antes': True,
  'eu estava ruim antes': False,
  'eu estava triste antes': False,
  'estou muito ruim agora': False,
  'isso é muito bom agora': True,
  'isso é muito triste agora': False,
  'isso era ruim antes': False,
  'isso era muito bom antes': True,
  'isso era muito ruim antes': False,
  'isso era muito feliz antes': True,
  'isso era muito triste antes': False,
  'eu estava bem e não ruim antes': True,
  'eu não estava bem e não estava feliz antes': False,
  'não estou nada ruim ou triste agora': True,
  'não estou nada bom ou feliz agora': False,
  'isso não era feliz e não era bom antes': False,
}


test_data = {
  'Estou feliz': True,
  'estou bem': True,
  'isso não é feliz': False,
  'não estou bom': False,
  'isso não é ruim': True,
  'não estou triste': True,
  'estou muito bom': True,
  'Estou muito ruim': False,
  'não estou muito triste': False,
  'Estou ruim, não bom': False,
  'Estou bom e feliz': True,
  'não estou bom e não estou feliz': False,
  'não estou nada triste': True,
  'isso não é nada bom': False,
  'isso não é nada ruim': True,
  'Estou bom agora': True,
  'Estou triste agora': False,
  'Estou muito ruim agora': False,
  'isso era bom antes': True,
  'eu não estava feliz e não estava bom antes': False,
}
