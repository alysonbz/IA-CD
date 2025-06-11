from src.utils import load_chocolate_dataset

chocolate = load_chocolate_dataset()
print(f'Dataset Shape: {chocolate.shape}')
print(chocolate.head())
print(chocolate.info())
print(chocolate.isnull().sum())