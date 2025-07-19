import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime

# 1. Carregar dados (xlsx)
df = pd.read_excel('Lojas.xlsx', engine='openpyxl')

# 2. Corrigir formato numérico brasileiro
cols_val = ['Faturamento médio (Mês)', 'Ticket Médio', 'Tamanho Loja']
for col in cols_val:
    df[col] = df[col].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

# 3. Criar faturamento por m2
df['Faturamento por m2'] = df['Faturamento médio (Mês)'] / df['Tamanho Loja']

# 4. Aplicar pesos antes da normalização

# Tamanho Loja e Faturamento por m2 ficam com peso 1 (normal)

# 5. Selecionar as colunas numéricas para normalizar
colunas_modelo = ['Faturamento médio (Mês)', 'Ticket Médio', 'Tamanho Loja', 'Faturamento por m2']
X_num = df[colunas_modelo]

# 6. Normalizar variáveis numéricas
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# 7. One-hot encode a variável UF
X_uf = pd.get_dummies(df['UF'], prefix='UF')

# 8. Concatenar variáveis numéricas normalizadas + variáveis dummies
import numpy as np
X_final = np.hstack([X_num_scaled, X_uf.values])

# 9. Aplicar clusterização
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df['Cluster'] = kmeans.fit_predict(X_final)

# 10. Resultado com identificador
resultado = df[['Cod', 'Loja', 'UF', 'Cluster']]

print(resultado)
