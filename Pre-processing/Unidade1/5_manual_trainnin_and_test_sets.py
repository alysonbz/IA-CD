from src.utils import load_volunteer_dataset

volunteer = load_volunteer_dataset()

def train_test_split(X,y,test_size,random_seed=1):
    
    np.random.seed(random_seed)
    
    #Junta X e y
    df = pd.concat([X, y], axis=1)
    
    train_list = []
    test_list = []
    
    for classe, grupo in df.groupby('category_desc'):
        
        #Embaralha os dados da classe
        grupo = grupo.sample(frac=1, random_state=random_seed)
        
        #Embaralha os dados para teste
        n_test = max(1,int(len(grupo) * test_size))
        
        #Divide os dados
        test = grupo.iloc[:n_test]
        train = grupo.iloc[n_test:]
        
        #Guardar os dados
        train_list.append(train)
        test_list.append(test)
    
    
    #Junta Tudo
    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)
    
    #Separa X e y nov.
    X_train = train_df.drop('category_desc', axis=1)
    y_train = train_df[['category_desc']]
    
    X_test = test_df.drop('category_desc', axis=1)
    y_test = test_df[['category_desc']]
    
    
    return X_train,X_test, y_train, y_test


# Exclua as colunas Latitude e Longitude de volunteer
volunteer_new = volunteer.drop(['Latitude', 'Longitude'], axis=1)

# Exclua as linhas com valores null da coluna category_desc de volunteer_new
volunteer = volunteer_new.dropna(subset=['category_desc'])

# mostre o balanceamento das classes em 'category_desc'
print(volunteer['category_desc'].value_counts(), '\n\n')

# Crie um DataFrame com todas as colunas, com exceção de ``category_desc``
X = volunteer.drop('category_desc', axis=1)

# Crie um dataframe de labels com a coluna category_desc
y = volunteer[['category_desc']]

# # Utiliza a a amostragem stratificada para separar o dataset em treino e teste
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size,random_seed=1)

# mostre o balanceamento das classes em 'category_desc' novamente
print(y_train['category_desc'].value_counts(), '\n\n')

print(y_test['category_desc'].value_counts())
