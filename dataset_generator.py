# pip install pandas numpy

import pandas as pd
import numpy as np
import random

# Função para gerar diagnósticos aleatórios
def gerar_diagnostico():
    return random.choice(['Depressão', 'TAG', 'TOC', 'Esquizofrenia', 'Transtorno Bipolar'])

# Mapeamento dos diagnósticos para índices
diagnostico_para_indice = {
    'Depressão': 0,
    'TAG': 1,
    'TOC': 2,
    'Esquizofrenia': 3,
    'Transtorno Bipolar': 4
}

# Função para gerar dados de biomarcadores aleatórios (como dados fictícios de fMRI)
def gerar_biomarcador():
    return round(np.random.uniform(0, 1), 3)

# Função para gerar escores aleatórios de escalas psicométricas
def gerar_escore_psicometrico(escala):
    return round(np.random.uniform(escala[0], escala[1]), 2)

# Configurações do dataset
num_pacientes = int(input("Insira a quantidade de pacientes que deseja na amostra: "))
idades = np.random.randint(18, 65, size=num_pacientes)
diagnosticos = [gerar_diagnostico() for _ in range(num_pacientes)]
fMRI = [gerar_biomarcador() for _ in range(num_pacientes)]  # Dados de biomarcadores (exemplo)
HAMD_scores = [gerar_escore_psicometrico((0, 52)) for _ in range(num_pacientes)]  # Escala Hamilton de Depressão (0 a 52)
BAI_scores = [gerar_escore_psicometrico((0, 63)) for _ in range(num_pacientes)]  # Escala de Ansiedade de Beck (0 a 63)

# Criando um DataFrame
df = pd.DataFrame({
    'Paciente_ID': range(1, num_pacientes + 1),
    'Idade': idades,
    'Diagnostico': diagnosticos,
    'Diagnostico_Indice': [diagnostico_para_indice[d] for d in diagnosticos],  # Índice do diagnóstico
    'Biomarcador_fMRI': fMRI,
    'HAMD_score': HAMD_scores,
    'BAI_score': BAI_scores
})

# Gerando o arquivo CSV
df.to_csv('dataset_pesquisa_transtornos.csv', index=False)

print("Arquivo CSV gerado com sucesso!")