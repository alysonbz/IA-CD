from carregar_dados import carregar_dataset
from exploracao_inicial import explorar_dados
from pre_processamento import preprocessar_dados
from separacao_dados import separar_dados

from norm_min_max import aplicar_min_max
from norm_z_score import aplicar_z_score
from norm_log import normalizacao_log

from treino_min_max import treino_knn_minmax
from treino_z_score import treinar_knn_zscore
from treino_log import treinar_knn_log

from validacao_cruzada import validacao_cruzada
from grid_search import executar_grid_search
from graficos import grafico_k

# Carregando dataset

df = carregar_dataset()

# Exploração inicial

explorar_dados(df)

# Pré-processamento

df = preprocessar_dados(df)

# Separação dos dados

X_train, X_test, y_train, y_test = separar_dados(df)

# Normalizações

X_train_minmax, X_test_minmax = aplicar_min_max(X_train, X_test)
X_train_z, X_test_z = aplicar_z_score(X_train, X_test)
X_train_log, X_test_log = normalizacao_log(X_train, X_test)

# Treinamentos

acc_minmax, k_minmax = treino_knn_minmax(
    X_train_minmax,
    X_test_minmax,
    y_train,
    y_test
)

acc_zscore, k_zscore = treinar_knn_zscore(
    X_train_z,
    X_test_z,
    y_train,
    y_test
)

acc_log, k_log = treinar_knn_log(
    X_train_log,
    X_test_log,
    y_train,
    y_test
)


# Gráfico

grafico_k()

# Validação cruzada

validacao_cruzada(X_train_z, y_train, k_zscore)

# Grid Search

executar_grid_search(X_train_z, y_train)